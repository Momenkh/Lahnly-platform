import {
  Controller,
  Get,
  Post,
  Delete,
  Param,
  Query,
  Body,
  UploadedFile,
  UseInterceptors,
  BadRequestException,
} from '@nestjs/common';
import {
  ApiTags,
  ApiBearerAuth,
  ApiOperation,
  ApiConsumes,
  ApiBody,
  ApiSecurity,
} from '@nestjs/swagger';
import { FileInterceptor } from '@nestjs/platform-express';
import { parseBuffer } from 'music-metadata';
import { SongsService } from './songs.service';
import { CreateSongDto } from './dto/create-song.dto';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import type { AuthUser } from '../common/types/auth-user';

const AUDIO_MIME_TYPES = [
  'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav',
  'audio/flac', 'audio/m4a', 'audio/mp4', 'audio/x-m4a',
  'audio/ogg', 'audio/webm',
];

@ApiTags('Songs')
@ApiBearerAuth()
@ApiSecurity('x-dev-user-id')
@Controller('songs')
export class SongsController {
  constructor(private readonly songsService: SongsService) {}

  @ApiOperation({ summary: 'Upload an audio file and create a song' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    schema: {
      type: 'object',
      required: ['file', 'title'],
      properties: {
        file: { type: 'string', format: 'binary', description: 'Audio file (mp3, wav, flac, m4a, ogg, webm)' },
        title: { type: 'string', example: 'Bohemian Rhapsody' },
      },
    },
  })
  @Post()
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 200 * 1024 * 1024 } }))
  async create(
    @CurrentUser() user: AuthUser,
    @Body() dto: CreateSongDto,
    @UploadedFile() file: Express.Multer.File,
  ) {
    if (!file) throw new BadRequestException('Audio file is required');
    if (!AUDIO_MIME_TYPES.includes(file.mimetype)) {
      throw new BadRequestException(`Unsupported audio format: ${file.mimetype}`);
    }

    const metadata = await parseBuffer(file.buffer, { mimeType: file.mimetype });
    const durationSeconds = metadata.format.duration;
    if (!durationSeconds || durationSeconds <= 0) {
      throw new BadRequestException('Could not detect audio duration — ensure the file is a valid audio file');
    }

    return this.songsService.create(user.id, dto, file, durationSeconds);
  }

  @ApiOperation({ summary: 'List all songs for the current user' })
  @Get()
  findAll(
    @CurrentUser() user: AuthUser,
    @Query('page') page = '1',
    @Query('limit') limit = '20',
  ) {
    return this.songsService.findAllForUser(user.id, +page, +limit);
  }

  @ApiOperation({ summary: 'Get a song by ID' })
  @Get(':id')
  findOne(@CurrentUser() user: AuthUser, @Param('id') id: string) {
    return this.songsService.findOne(user.id, id);
  }

  @ApiOperation({ summary: 'Delete a song and all its results' })
  @Delete(':id')
  remove(@CurrentUser() user: AuthUser, @Param('id') id: string) {
    return this.songsService.remove(user.id, id);
  }
}
