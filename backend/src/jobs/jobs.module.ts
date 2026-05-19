import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bull';
import { JobsController } from './jobs.controller';
import { JobsService } from './jobs.service';
import { SongsModule } from '../songs/songs.module';
import { ResultsModule } from '../results/results.module';
import { TRANSCRIPTION_QUEUE } from '../queue/transcription.types';

@Module({
  imports: [
    BullModule.registerQueue({ name: TRANSCRIPTION_QUEUE }),
    SongsModule,
    ResultsModule,
  ],
  controllers: [JobsController],
  providers: [JobsService],
})
export class JobsModule {}
