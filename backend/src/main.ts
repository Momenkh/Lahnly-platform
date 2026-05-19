import 'reflect-metadata';
import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { rawBody: true });

  app.setGlobalPrefix('api');
  app.enableCors();
  app.useGlobalPipes(
    new ValidationPipe({ whitelist: true, forbidNonWhitelisted: true, transform: true }),
  );

  const port = process.env.PORT ?? 3000;

  const config = new DocumentBuilder()
    .setTitle('Lahnly API')
    .setDescription(
      'Music transcription platform API\n\n' +
      '**Dev mode:** Set `DEV_BYPASS=true` in `.env` and use the `x-dev-user-id` API key ' +
      'below instead of a Bearer token. Any string (e.g. `dev-user-1`) becomes your user ID.',
    )
    .setVersion('1.0')
    .addServer(`http://localhost:${port}`, 'Local')
    .addBearerAuth()
    .addApiKey({ type: 'apiKey', in: 'header', name: 'x-dev-user-id' }, 'x-dev-user-id')
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('docs', app, document);

  await app.listen(port);
  console.log(`Backend listening on http://localhost:${port}/api`);
  console.log(`Swagger docs at  http://localhost:${port}/docs`);
}
bootstrap();
