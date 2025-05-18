export interface DeepfakeResult {
  id: string;
  imageUrl: string;
  confidence: number;
  isSelected: boolean;
  timestamp: string;
  celebrity?: string;
  images?: string[];
  youtube_videos?: string[];
  vimeo_videos?: string[];
  dailymotion_videos?: string[];
} 