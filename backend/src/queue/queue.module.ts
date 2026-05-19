import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bull';
import { TranscriptionProcessor } from './transcription.processor';
import { TRANSCRIPTION_QUEUE } from './transcription.types';

@Module({
  imports: [BullModule.registerQueue({ name: TRANSCRIPTION_QUEUE })],
  providers: [TranscriptionProcessor],
})
export class QueueModule {}
