import { Controller, Get, Post, Delete, Param, Body, Query } from '@nestjs/common';
import { ApiTags, ApiBearerAuth, ApiOperation, ApiSecurity } from '@nestjs/swagger';
import { PublicationsService } from './publications.service';
import { CreatePublicationDto } from './dto/create-publication.dto';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { Public } from '../common/decorators/public.decorator';
import type { AuthUser } from '../common/types/auth-user';

@ApiTags('Publications')
@Controller('publications')
export class PublicationsController {
  constructor(private readonly publicationsService: PublicationsService) {}

  @ApiOperation({ summary: 'Publish a completed song publicly' })
  @ApiBearerAuth()
  @ApiSecurity('x-dev-user-id')
  @Post()
  publish(@CurrentUser() user: AuthUser, @Body() dto: CreatePublicationDto) {
    return this.publicationsService.publish(user.id, dto);
  }

  @ApiOperation({ summary: 'Unpublish a song' })
  @ApiBearerAuth()
  @ApiSecurity('x-dev-user-id')
  @Delete(':id')
  unpublish(@CurrentUser() user: AuthUser, @Param('id') id: string) {
    return this.publicationsService.unpublish(user.id, id);
  }

  @ApiOperation({ summary: 'Browse public songs (no auth required)' })
  @Public()
  @Get()
  findAll(
    @Query('page') page = '1',
    @Query('limit') limit = '20',
    @Query('search') search?: string,
  ) {
    return this.publicationsService.findAll(+page, +limit, search);
  }

  @ApiOperation({ summary: 'Get a published song with preview URLs (no auth required)' })
  @Public()
  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.publicationsService.findOne(id);
  }
}
