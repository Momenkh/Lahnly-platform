import { ApiProperty } from '@nestjs/swagger';
import { IsArray } from 'class-validator';

export class UpdateNotesDto {
  @ApiProperty({
    description: 'Full replacement notes array',
    type: 'array',
    items: { type: 'object' },
  })
  @IsArray()
  notes: object[];
}
