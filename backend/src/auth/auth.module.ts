import { Module } from '@nestjs/common';
import { APP_GUARD } from '@nestjs/core';
import { ClerkGuard } from './clerk.guard';

@Module({
  providers: [
    {
      provide: APP_GUARD,
      useClass: ClerkGuard,
    },
  ],
})
export class AuthModule {}
