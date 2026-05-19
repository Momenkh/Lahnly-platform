import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsString, IsNotEmpty, IsOptional, IsArray, IsIn, ArrayMinSize } from 'class-validator';

const VALID_INSTRUMENTS = ['guitar', 'bass', 'piano', 'vocals', 'drums'];

export class CreatePublicationDto {
  @ApiProperty({ example: 'clx123abc' })
  @IsString()
  @IsNotEmpty()
  songId: string;

  @ApiProperty({ example: 'Bohemian Rhapsody' })
  @IsString()
  @IsNotEmpty()
  title: string;

  @ApiPropertyOptional({ example: 'A classic Queen track' })
  @IsOptional()
  @IsString()
  description?: string;

  @ApiProperty({
    example: ['guitar', 'bass'],
    enum: VALID_INSTRUMENTS,
    isArray: true,
  })
  @IsArray()
  @ArrayMinSize(1)
  @IsIn(VALID_INSTRUMENTS, { each: true })
  instruments: string[];
}
