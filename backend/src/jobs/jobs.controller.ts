import { Controller, Get, Post, Param, Body } from '@nestjs/common';
import { ApiTags, ApiBearerAuth, ApiOperation, ApiSecurity } from '@nestjs/swagger';
import { JobsService } from './jobs.service';
import { ResultsService } from '../results/results.service';
import { CreateJobDto } from './dto/create-job.dto';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import type { AuthUser } from '../common/types/auth-user';

@ApiTags('Jobs')
@ApiBearerAuth()
@ApiSecurity('x-dev-user-id')
@Controller()
export class JobsController {
  constructor(
    private readonly jobsService: JobsService,
    private readonly resultsService: ResultsService,
  ) {}

  @ApiOperation({ summary: 'Create a processing job (deducts tokens)' })
  @Post('songs/:songId/jobs')
  create(
    @CurrentUser() user: AuthUser,
    @Param('songId') songId: string,
    @Body() dto: CreateJobDto,
  ) {
    return this.jobsService.create(user.id, songId, dto);
  }

  @ApiOperation({ summary: 'List all jobs for a song' })
  @Get('songs/:songId/jobs')
  findAllForSong(@CurrentUser() user: AuthUser, @Param('songId') songId: string) {
    return this.jobsService.findAllForSong(user.id, songId);
  }

  @ApiOperation({ summary: 'Get job status and results' })
  @Get('jobs/:id')
  findOne(@CurrentUser() user: AuthUser, @Param('id') id: string) {
    return this.jobsService.findOne(user.id, id);
  }

  @ApiOperation({ summary: 'List all available stems (instrument results) for a job' })
  @Get('jobs/:jobId/results')
  listResults(@CurrentUser() user: AuthUser, @Param('jobId') jobId: string) {
    return this.resultsService.listForJob(user.id, jobId);
  }
}
