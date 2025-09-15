import React from 'react';
import { Box, Typography } from '@mui/material';

/**
 * Highlights matching terms in text based on a search query
 */
export const highlightText = (
  text: string,
  query: string,
  variant: 'body2' | 'body1' | 'subtitle2' = 'body2'
): React.ReactElement => {
  if (!query || !text) {
    return <Typography variant={variant}>{text}</Typography>;
  }

  const queryLower = query.toLowerCase().trim();
  const textLower = text.toLowerCase();
  
  // First, try to match the entire phrase (highest priority)
  if (textLower.includes(queryLower)) {
    const phrasePattern = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(phrasePattern);
    
    return (
      <Typography variant={variant}>
        {parts.map((part, index) => {
          const isPhraseMatch = part.toLowerCase() === queryLower;
          
          if (isPhraseMatch) {
            return (
              <Box
                key={index}
                component="span"
                sx={{
                  backgroundColor: '#ffeb3b', // Yellow for all matches
                  color: '#000',
                  padding: '2px 4px',
                  borderRadius: '3px',
                  fontWeight: 'bold',
                }}
              >
                {part}
              </Box>
            );
          }
          
          return part;
        })}
      </Typography>
    );
  }
  
  // If no phrase match, try whole words (medium priority)
  const allQueryWords = queryLower.split(/\s+/).filter(word => word.length > 0);
  const meaningfulWords = allQueryWords.filter(word => 
    word.length > 1 && 
    !['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'].includes(word)
  );
  const queryWords = meaningfulWords.length > 0 ? meaningfulWords : allQueryWords;
  
  if (queryWords.length === 0) {
    return <Typography variant={variant}>{text}</Typography>;
  }

  // Create a regex pattern that matches words (including partial matches like "girl" in "girlfriend")
  const wordPattern = new RegExp(`(${queryWords.map(word => 
    word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  ).join('|')})`, 'gi');
  
  const parts = text.split(wordPattern);
  
  return (
    <Typography variant={variant}>
      {parts.map((part, index) => {
        const isWordMatch = queryWords.some(word => 
          part.toLowerCase() === word.toLowerCase()
        );
        
        if (isWordMatch) {
          return (
            <Box
              key={index}
              component="span"
              sx={{
                backgroundColor: '#ffeb3b', // Yellow for word matches (medium priority)
                color: '#000',
                padding: '2px 4px',
                borderRadius: '3px',
                fontWeight: 'bold',
              }}
            >
              {part}
            </Box>
          );
        }
        
        return part;
      })}
    </Typography>
  );
};

/**
 * Highlights matching terms in lyrics with line breaks preserved
 */
export const highlightLyrics = (
  lyrics: string,
  query: string
): React.ReactElement => {
  if (!query || !lyrics) {
    return <Typography variant="body2">{lyrics}</Typography>;
  }

  const queryLower = query.toLowerCase().trim();
  const lyricsLower = lyrics.toLowerCase();
  
  // First, try to match the entire phrase (highest priority)
  if (lyricsLower.includes(queryLower)) {
    const phrasePattern = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = lyrics.split(phrasePattern);
    
    return (
      <Typography
        variant="body2"
        sx={{
          whiteSpace: 'pre-line',
          fontFamily: 'monospace',
          fontSize: '0.9rem',
          lineHeight: 1.4,
        }}
      >
        {parts.map((part, index) => {
          const isPhraseMatch = part.toLowerCase() === queryLower;
          
          if (isPhraseMatch) {
            return (
              <Box
                key={index}
                component="span"
                sx={{
                  backgroundColor: '#ffeb3b', // Yellow for all matches
                  color: '#000',
                  padding: '2px 4px',
                  borderRadius: '3px',
                  fontWeight: 'bold',
                }}
              >
                {part}
              </Box>
            );
          }
          
          return part;
        })}
      </Typography>
    );
  }
  
  // If no phrase match, try whole words (medium priority)
  const allQueryWords = queryLower.split(/\s+/).filter(word => word.length > 0);
  const meaningfulWords = allQueryWords.filter(word => 
    word.length > 1 && 
    !['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'].includes(word)
  );
  const queryWords = meaningfulWords.length > 0 ? meaningfulWords : allQueryWords;
  
  if (queryWords.length === 0) {
    return <Typography variant="body2">{lyrics}</Typography>;
  }

  // Create a regex pattern that matches words (including partial matches like "girl" in "girlfriend")
  const wordPattern = new RegExp(`(${queryWords.map(word => 
    word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  ).join('|')})`, 'gi');
  
  const parts = lyrics.split(wordPattern);
  
  return (
    <Typography
      variant="body2"
      sx={{
        whiteSpace: 'pre-line',
        fontFamily: 'monospace',
        fontSize: '0.9rem',
        lineHeight: 1.4,
      }}
    >
      {parts.map((part, index) => {
        const isWordMatch = queryWords.some(word => 
          part.toLowerCase() === word.toLowerCase()
        );
        
        if (isWordMatch) {
          return (
            <Box
              key={index}
              component="span"
              sx={{
                backgroundColor: '#ffeb3b', // Yellow for word matches (medium priority)
                color: '#000',
                padding: '2px 4px',
                borderRadius: '3px',
                fontWeight: 'bold',
              }}
            >
              {part}
            </Box>
          );
        }
        
        return part;
      })}
    </Typography>
  );
};
