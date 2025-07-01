import json
import streamlit as st
import requests
import base64
import io
import os
import re
import wave
from typing import Any, Optional, List

st.set_page_config(
    page_title="Transcript to Audio Converter",
    page_icon="🎧",
    layout="wide"
)

def convert_text_to_audio(text: str, api_key: str) -> Optional[bytes]:
    """Convert text to audio using Deepgram's text-to-speech API."""
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "text": text
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Error from Deepgram API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling Deepgram API: {str(e)}")
        return None

def chunk_text(text: str, max_chars: int = 1800) -> List[str]:
    """Split text into chunks that fit within API character limits."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # If a single sentence is too long, split it by words
                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = word
                        else:
                            # If a single word is too long, truncate it
                            chunks.append(word[:max_chars])
                    else:
                        current_chunk += " " + word if current_chunk else word
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def merge_audio_chunks(audio_chunks: List[bytes]) -> bytes:
    """Merge multiple audio chunks into a single audio file."""
    if not audio_chunks:
        return b""
    
    if len(audio_chunks) == 1:
        return audio_chunks[0]
    
    # For simplicity, concatenate raw audio data
    # This works for WAV files from Deepgram
    merged_audio = b""
    
    for i, chunk in enumerate(audio_chunks):
        if i == 0:
            # Keep the full header for the first chunk
            merged_audio = chunk
        else:
            # Skip the WAV header (44 bytes) for subsequent chunks
            if len(chunk) > 44:
                merged_audio += chunk[44:]
    
    return merged_audio

def format_transcript_to_text(transcripts_str: str) -> str:
    """Format the transcript text for natural speech synthesis."""
    try:
        transcripts: list[dict[str, Any]] = json.loads(transcripts_str)
        result = ""
        for item in transcripts:
            text = item.get("text", "")
            speaker = item.get("speaker", "")
            
            # Add speaker identification for dialogue
            if speaker:
                result += f"{speaker}: {text}\n\n"
            else:
                result += f"{text}\n\n"
        
        formatted_text = result.strip()
    except json.JSONDecodeError:
        # If it's not JSON, assume it's plain text
        formatted_text = transcripts_str.strip()
    
    # Apply natural speech formatting
    return enhance_text_for_speech(formatted_text)

def enhance_text_for_speech(text: str) -> str:
    """Enhance text with proper punctuation and formatting for natural speech."""
    # Split into paragraphs to handle speaker changes
    paragraphs = text.split('\n\n')
    enhanced_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Check if this is speaker dialogue (contains "Speaker:" or similar)
        speaker_match = re.match(r'^([^:]+):\s*(.+)$', paragraph.strip(), re.DOTALL)
        
        if speaker_match:
            speaker_name = speaker_match.group(1).strip()
            speech_text = speaker_match.group(2).strip()
            
            # Format speaker name for natural pronunciation
            formatted_speaker = format_speaker_name(speaker_name)
            enhanced_speech = format_speech_text(speech_text)
            
            enhanced_paragraphs.append(f"{formatted_speaker}: {enhanced_speech}")
        else:
            # Regular text without speaker identification
            enhanced_paragraphs.append(format_speech_text(paragraph.strip()))
    
    return '\n\n'.join(enhanced_paragraphs)

def format_speaker_name(speaker_name: str) -> str:
    """Format speaker names for natural pronunciation."""
    # Handle common speaker patterns
    speaker_name = re.sub(r'Speaker\s*(\d+)', r'Speaker \1', speaker_name, flags=re.IGNORECASE)
    speaker_name = re.sub(r'User\s*(\d+)', r'User \1', speaker_name, flags=re.IGNORECASE)
    
    # Add comma after speaker name for natural pause
    return speaker_name

def format_speech_text(text: str) -> str:
    """Format speech text with proper punctuation for natural pauses."""
    if not text.strip():
        return text
    
    # Ensure sentences end with proper punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    formatted_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add natural pauses with commas
        sentence = add_natural_commas(sentence)
        
        # Ensure sentence ends with punctuation
        if not re.search(r'[.!?]$', sentence):
            # Determine appropriate ending punctuation
            if re.search(r'\b(what|when|where|why|how|who|is|are|do|does|did|can|could|would|will|should)\b', sentence.lower()):
                sentence += '?'
            elif re.search(r'\b(wow|great|amazing|excellent|fantastic|oh|ah|yes|no)\b', sentence.lower()):
                sentence += '!'
            else:
                sentence += '.'
        
        formatted_sentences.append(sentence)
    
    result = ' '.join(formatted_sentences)
    
    # Add strategic hyphens for additional pauses in long sentences
    result = add_strategic_pauses(result)
    
    return result

def add_natural_commas(text: str) -> str:
    """Add commas for natural speech flow."""
    # Add commas before names in direct address
    text = re.sub(r'\b(hi|hello|hey|dear|mr|mrs|ms|dr)\s+([A-Z][a-z]+)', r'\1, \2', text, flags=re.IGNORECASE)
    
    # Add commas before conjunctions in compound sentences
    text = re.sub(r'(\w+)\s+(and|or)\s+(\w+)', r'\1, \2 \3', text)
    
    # Add commas after introductory words
    text = re.sub(r'^(well|so|now|then|actually|basically|essentially|however|meanwhile|furthermore|therefore|moreover)\s+', r'\1, ', text, flags=re.IGNORECASE)
    
    # Add commas before conjunctions in longer sentences (avoid double commas)
    text = re.sub(r'(?<!,)\s+(but|yet|so)\s+(?=\w{3,})', r', \1 ', text)
    
    return text

def add_strategic_pauses(text: str) -> str:
    """Add hyphens for strategic pauses in speech."""
    # Add pause after transition words
    text = re.sub(r'\b(however|therefore|meanwhile|furthermore|moreover|consequently|additionally|finally)\b', r'\1 --', text, flags=re.IGNORECASE)
    
    # Add pause before important information
    text = re.sub(r'\b(important|note|remember|please|understand)\b', r'-- \1', text, flags=re.IGNORECASE)
    
    return text


def main():
    st.title("🎧 Transcript to Audio Converter")
    st.markdown("Convert your call transcripts to high-quality audio using Deepgram's text-to-speech technology.")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Deepgram API Key",
            type="password",
            help="Enter your Deepgram API key. Get one at https://deepgram.com/"
        )
        
        if not api_key:
            st.warning("Please enter your Deepgram API key to continue.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Transcript")
        transcript = st.text_area(
            "Call Transcript",
            height=300,
            placeholder="Enter your call transcript here...",
            help="Paste the transcript of your call that you want to convert to audio."
        )
        
        # Voice model selection
        voice_model = st.selectbox(
            "Select Voice Model",
            options=[
                "aura-asteria-en",
                "aura-luna-en",
                "aura-stella-en",
                "aura-athena-en",
                "aura-hera-en",
                "aura-orion-en",
                "aura-arcas-en",
                "aura-perseus-en",
                "aura-angus-en",
                "aura-orpheus-en",
                "aura-helios-en",
                "aura-zeus-en"
            ],
            index=0,
            help="Choose the voice model for text-to-speech conversion."
        )
        
        convert_button = st.button("🎵 Convert to Audio", type="primary", use_container_width=True)
    
    with col2:
        st.header("🔊 Generated Audio")
        
        if convert_button:
            if not api_key:
                st.error("Please enter your Deepgram API key in the sidebar.")
            elif not transcript.strip():
                st.error("Please enter a transcript to convert.")
            else:
                with st.spinner("Converting transcript to audio..."):
                    transcript_text = format_transcript_to_text(transcript)
                    text_chunks = chunk_text(transcript_text)
                    
                    if len(text_chunks) > 1:
                        st.info(f"Text is long, processing in {len(text_chunks)} chunks...")
                    
                    audio_chunks = []
                    progress_bar = st.progress(0)
                    
                    for i, chunk in enumerate(text_chunks):
                        url = f"https://api.deepgram.com/v1/speak?model={voice_model}"
                        
                        headers = {
                            "Authorization": f"Token {api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        payload = {
                            "text": chunk
                        }
                        
                        try:
                            response = requests.post(url, json=payload, headers=headers)
                            
                            if response.status_code == 200:
                                audio_chunks.append(response.content)
                                progress_bar.progress((i + 1) / len(text_chunks))
                            else:
                                st.error(f"Error from Deepgram API (chunk {i+1}): {response.status_code} - {response.text}")
                                break
                                
                        except Exception as e:
                            st.error(f"Error calling Deepgram API (chunk {i+1}): {str(e)}")
                            break
                    
                    if len(audio_chunks) == len(text_chunks):
                        # Merge audio chunks
                        audio_data = merge_audio_chunks(audio_chunks)
                        
                        # Display audio player
                        st.audio(audio_data, format="audio/wav")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_data,
                            file_name="transcript_audio.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
                        
                        st.success("✅ Audio generated successfully!")
                    else:
                        st.error("Failed to generate audio for all text chunks.")

    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Get a Deepgram API Key**: Sign up at [deepgram.com](https://deepgram.com) and get your API key
        2. **Enter API Key**: Paste your API key in the sidebar
        3. **Input Transcript**: Paste your call transcript in the text area
        4. **Select Voice**: Choose your preferred voice model
        5. **Convert**: Click the "Convert to Audio" button
        6. **Listen & Download**: Play the audio or download the file
        """)

if __name__ == "__main__":
    main()