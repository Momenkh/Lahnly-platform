export const TRANSCRIPTION_QUEUE = 'transcription';

export interface TranscriptionJobPayload {
  processingJobId: string;
  songId: string;
  userId: string;
  audioStorageKey: string;
  instruments: string[];
  durationSeconds: number;
}
