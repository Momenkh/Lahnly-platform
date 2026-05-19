import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsNotEmpty } from 'class-validator';

export class CreateSongDto {
  @ApiProperty({ example: 'Bohemian Rhapsody' })
  @IsString()
  @IsNotEmpty()
  title: string;
}
