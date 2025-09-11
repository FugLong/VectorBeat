import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { SearchState, TrackFilters, SearchMode } from '../types';

// Action types
type SearchAction =
  | { type: 'SET_QUERY'; payload: string }
  | { type: 'SET_FILTERS'; payload: TrackFilters }
  | { type: 'SET_RESULTS'; payload: any[] }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_MODE'; payload: SearchMode }
  | { type: 'SET_PAGE'; payload: number }
  | { type: 'SET_HAS_MORE'; payload: boolean }
  | { type: 'CLEAR_RESULTS' }
  | { type: 'RESET_SEARCH' };

// Initial state
const initialState: SearchState = {
  query: '',
  filters: {},
  results: [],
  loading: false,
  error: null,
  mode: 'metadata',
  page: 1,
  hasMore: false,
};

// Reducer
function searchReducer(state: SearchState, action: SearchAction): SearchState {
  switch (action.type) {
    case 'SET_QUERY':
      return { ...state, query: action.payload };
    case 'SET_FILTERS':
      return { ...state, filters: action.payload };
    case 'SET_RESULTS':
      return { ...state, results: action.payload };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'SET_MODE':
      return { ...state, mode: action.payload };
    case 'SET_PAGE':
      return { ...state, page: action.payload };
    case 'SET_HAS_MORE':
      return { ...state, hasMore: action.payload };
    case 'CLEAR_RESULTS':
      return { ...state, results: [], page: 1, hasMore: false };
    case 'RESET_SEARCH':
      return initialState;
    default:
      return state;
  }
}

// Context
const SearchContext = createContext<{
  state: SearchState;
  dispatch: React.Dispatch<SearchAction>;
} | null>(null);

// Provider component
export function SearchProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(searchReducer, initialState);

  return (
    <SearchContext.Provider value={{ state, dispatch }}>
      {children}
    </SearchContext.Provider>
  );
}

// Hook to use search context
export function useSearch() {
  const context = useContext(SearchContext);
  if (!context) {
    throw new Error('useSearch must be used within a SearchProvider');
  }
  return context;
}
