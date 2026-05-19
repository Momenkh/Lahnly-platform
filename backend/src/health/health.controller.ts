import { Controller, Get } from '@nestjs/common';
import { Public } from '../common/decorators/public.decorator';
import { DatabaseService } from '../prisma/database.service';

@Public()
@Controller('health')
export class HealthController {
  constructor(private db: DatabaseService) {}

  @Get()
  async check() {
    await this.db.query('SELECT 1');
    return { status: 'ok' };
  }
}
