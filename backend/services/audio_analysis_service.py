"""
Audio analysis service for extracting musical features from audio embeddings.
Provides tempo, mood, and instrument analysis using ML models.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import librosa
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioAnalysisService:
    """Service for analyzing audio features and extracting musical characteristics."""
    
    def __init__(self):
        """Initialize the audio analysis service."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
    
    def analyze_audio_file(self, audio_path: str) -> Dict[str, any]:
        """Analyze an audio file and extract musical features."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Extract features
            tempo = self._extract_tempo(audio, sr)
            mood = self._analyze_mood(audio, sr)
            instruments = self._detect_instruments(audio, sr)
            energy = self._calculate_energy(audio)
            valence = self._calculate_valence(audio, sr)
            
            return {
                'tempo': tempo,
                'mood': mood,
                'instruments': instruments,
                'energy': energy,
                'valence': valence,
                'duration': len(audio) / sr
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze audio file {audio_path}: {str(e)}")
            return self._get_default_analysis()
    
    def analyze_audio_array(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, any]:
        """Analyze audio array and extract musical features."""
        try:
            # Resample if necessary
            if sample_rate != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Extract features
            tempo = self._extract_tempo(audio, self.sample_rate)
            mood = self._analyze_mood(audio, self.sample_rate)
            instruments = self._detect_instruments(audio, self.sample_rate)
            energy = self._calculate_energy(audio)
            valence = self._calculate_valence(audio, self.sample_rate)
            
            return {
                'tempo': tempo,
                'mood': mood,
                'instruments': instruments,
                'energy': energy,
                'valence': valence,
                'duration': len(audio) / self.sample_rate
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze audio array: {str(e)}")
            return self._get_default_analysis()
    
    def _extract_tempo(self, audio: np.ndarray, sr: int) -> Dict[str, any]:
        """Extract tempo and rhythm information."""
        try:
            # Get tempo using librosa
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            
            # Classify tempo
            if tempo < 60:
                tempo_category = "Very Slow"
            elif tempo < 80:
                tempo_category = "Slow"
            elif tempo < 100:
                tempo_category = "Moderate"
            elif tempo < 120:
                tempo_category = "Upbeat"
            elif tempo < 140:
                tempo_category = "Fast"
            else:
                tempo_category = "Very Fast"
            
            # Calculate rhythm complexity
            rhythm_complexity = self._calculate_rhythm_complexity(audio, sr, beats)
            
            return {
                'bpm': round(tempo, 1),
                'category': tempo_category,
                'rhythm_complexity': rhythm_complexity,
                'beat_strength': float(np.mean(beats)) if len(beats) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to extract tempo: {str(e)}")
            return {'bpm': 120.0, 'category': 'Moderate', 'rhythm_complexity': 'Medium', 'beat_strength': 0.5}
    
    def _analyze_mood(self, audio: np.ndarray, sr: int) -> str:
        """Analyze the mood/emotional character of the audio."""
        try:
            # Extract features for mood analysis
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Calculate mood indicators
            brightness = float(np.mean(spectral_centroids))
            roughness = float(np.mean(zero_crossing_rate))
            energy = float(np.mean(librosa.feature.rms(y=audio)))
            
            # Classify mood based on features
            if energy > 0.1 and brightness > 2000:
                if roughness > 0.1:
                    return "Energetic"
                else:
                    return "Happy"
            elif energy < 0.05 and brightness < 1500:
                if roughness < 0.05:
                    return "Calm"
                else:
                    return "Melancholic"
            elif brightness > 2000 and energy < 0.08:
                return "Dreamy"
            elif roughness > 0.15:
                return "Aggressive"
            else:
                return "Neutral"
                
        except Exception as e:
            logger.error(f"Failed to analyze mood: {str(e)}")
            return "Unknown"
    
    def _detect_instruments(self, audio: np.ndarray, sr: int) -> List[str]:
        """Detect likely instruments in the audio."""
        try:
            instruments = []
            
            # Extract harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Analyze harmonic content
            harmonic_energy = float(np.mean(librosa.feature.rms(y=harmonic)))
            percussive_energy = float(np.mean(librosa.feature.rms(y=percussive)))
            
            # Detect drums/percussion
            if percussive_energy > harmonic_energy * 0.5:
                instruments.append("Drums")
            
            # Analyze spectral features for different instruments
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            
            avg_centroid = float(np.mean(spectral_centroids))
            avg_bandwidth = float(np.mean(spectral_bandwidth))
            
            # Guitar detection (mid-range frequencies, moderate bandwidth)
            if 1000 < avg_centroid < 3000 and 500 < avg_bandwidth < 2000:
                instruments.append("Guitar")
            
            # Bass detection (low frequencies)
            if avg_centroid < 1000:
                instruments.append("Bass")
            
            # Piano detection (wide frequency range, harmonic content)
            if avg_bandwidth > 2000 and harmonic_energy > 0.05:
                instruments.append("Piano")
            
            # Vocals detection (human voice range)
            if 200 < avg_centroid < 2000 and harmonic_energy > 0.03:
                instruments.append("Vocals")
            
            # Synthesizer detection (electronic characteristics)
            if avg_bandwidth > 3000 and harmonic_energy > 0.08:
                instruments.append("Synthesizer")
            
            # If no instruments detected, add defaults
            if not instruments:
                instruments = ["Guitar", "Bass", "Drums", "Vocals"]
            
            return instruments
            
        except Exception as e:
            logger.error(f"Failed to detect instruments: {str(e)}")
            return ["Guitar", "Bass", "Drums", "Vocals"]
    
    def _calculate_energy(self, audio: np.ndarray) -> float:
        """Calculate the energy/intensity of the audio."""
        try:
            rms = librosa.feature.rms(y=audio)
            return float(np.mean(rms))
        except Exception as e:
            logger.error(f"Failed to calculate energy: {str(e)}")
            return 0.5
    
    def _calculate_valence(self, audio: np.ndarray, sr: int) -> float:
        """Calculate the valence (positive/negative emotion) of the audio."""
        try:
            # Use spectral features to estimate valence
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            
            # Higher centroids and rolloff typically indicate more positive valence
            centroid_valence = min(1.0, float(np.mean(spectral_centroids)) / 3000)
            rolloff_valence = min(1.0, float(np.mean(spectral_rolloff)) / 5000)
            
            return (centroid_valence + rolloff_valence) / 2
            
        except Exception as e:
            logger.error(f"Failed to calculate valence: {str(e)}")
            return 0.5
    
    def _calculate_rhythm_complexity(self, audio: np.ndarray, sr: int, beats: np.ndarray) -> str:
        """Calculate the complexity of the rhythm."""
        try:
            if len(beats) < 2:
                return "Simple"
            
            # Calculate beat intervals
            beat_intervals = np.diff(beats)
            interval_variance = float(np.var(beat_intervals))
            
            if interval_variance < 0.1:
                return "Simple"
            elif interval_variance < 0.3:
                return "Medium"
            else:
                return "Complex"
                
        except Exception as e:
            logger.error(f"Failed to calculate rhythm complexity: {str(e)}")
            return "Medium"
    
    def _get_default_analysis(self) -> Dict[str, any]:
        """Return default analysis when extraction fails."""
        return {
            'tempo': {'bpm': 120.0, 'category': 'Moderate', 'rhythm_complexity': 'Medium', 'beat_strength': 0.5},
            'mood': 'Unknown',
            'instruments': ['Guitar', 'Bass', 'Drums', 'Vocals'],
            'energy': 0.5,
            'valence': 0.5,
            'duration': 0.0
        }
    
    def format_tempo_description(self, tempo_data: Dict[str, any]) -> str:
        """Format tempo data into a natural language description."""
        bpm = tempo_data['bpm']
        category = tempo_data['category']
        complexity = tempo_data['rhythm_complexity']
        
        return f"{category.lower()} tempo at {bpm} BPM with {complexity.lower()} rhythm"
    
    def format_mood_description(self, mood: str, energy: float, valence: float) -> str:
        """Format mood data into a natural language description."""
        energy_desc = "high energy" if energy > 0.7 else "low energy" if energy < 0.3 else "moderate energy"
        valence_desc = "positive" if valence > 0.6 else "negative" if valence < 0.4 else "neutral"
        
        return f"{mood.lower()} mood with {energy_desc} and {valence_desc} emotion"
    
    def format_instruments_description(self, instruments: List[str]) -> str:
        """Format instruments list into a natural language description."""
        if not instruments:
            return "Unknown instruments"
        
        if len(instruments) == 1:
            return f"featuring {instruments[0].lower()}"
        elif len(instruments) == 2:
            return f"featuring {instruments[0].lower()} and {instruments[1].lower()}"
        else:
            main_instruments = instruments[:2]
            other_count = len(instruments) - 2
            return f"featuring {', '.join(main_instruments).lower()} and {other_count} other instruments"
