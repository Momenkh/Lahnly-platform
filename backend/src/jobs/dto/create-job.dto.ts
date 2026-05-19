import { ApiProperty } from '@nestjs/swagger';
import { IsArray, IsIn, ArrayMinSize, ArrayMaxSize } from 'class-validator';

const VALID_INSTRUMENTS = ['guitar', 'bass', 'piano', 'vocals', 'drums'];

export class CreateJobDto {
  @ApiProperty({
    example: ['guitar', 'bass'],
    description: 'Instruments to transcribe. Cost: ceil(duration_minutes) × instruments.length tokens.',
    enum: VALID_INSTRUMENTS,
    isArray: true,
  })
  @IsArray()
  @ArrayMinSize(1)
  @ArrayMaxSize(5)
  @IsIn(VALID_INSTRUMENTS, { each: true })
  instruments: string[];
}
