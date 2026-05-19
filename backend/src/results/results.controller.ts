import { Controller, Get, Patch, Param, Body } from '@nestjs/common';
import { ApiTags, ApiBearerAuth, ApiOperation, ApiSecurity } from '@nestjs/swagger';
import { ResultsService } from './results.service';
import { UpdateNotesDto } from './dto/update-notes.dto';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import type { AuthUser } from '../common/types/auth-user';

@ApiTags('Results')
@ApiBearerAuth()
@ApiSecurity('x-dev-user-id')
@Controller('jobs/:jobId/results/:instrument')
export class ResultsController {
  constructor(private readonly resultsService: ResultsService) {}

  @ApiOperation({ summary: 'Get notes for an instrument (latest revision or original)' })
  @Get('notes')
  getNotes(
    @CurrentUser() user: AuthUser,
    @Param('jobId') jobId: string,
    @Param('instrument') instrument: string,
  ) {
    return this.resultsService.getNotes(user.id, jobId, instrument);
  }

  @ApiOperation({ summary: 'Save edited notes (creates a new revision)' })
  @Patch('notes')
  saveNotes(
    @CurrentUser() user: AuthUser,
    @Param('jobId') jobId: string,
    @Param('instrument') instrument: string,
    @Body() dto: UpdateNotesDto,
  ) {
    return this.resultsService.saveNotes(user.id, jobId, instrument, dto);
  }

  @ApiOperation({ summary: 'Get a presigned URL for the preview WAV' })
  @Get('preview')
  getPreviewUrl(
    @CurrentUser() user: AuthUser,
    @Param('jobId') jobId: string,
    @Param('instrument') instrument: string,
  ) {
    return this.resultsService.getPreviewUrl(user.id, jobId, instrument);
  }
}
