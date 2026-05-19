import * as dotenv from 'dotenv';
dotenv.config();

import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient({ datasourceUrl: process.env.DATABASE_URL });

async function main() {
  const user = await prisma.user.upsert({
    where: { clerkId: 'dev-user-1' },
    update: {},
    create: {
      clerkId: 'dev-user-1',
      email: 'dev@lahnly.local',
      displayName: 'Dev User',
      tokenBalance: { create: { balance: 500 } },
    },
    include: { tokenBalance: true },
  });

  console.log(`Seeded user: ${user.email} (id: ${user.id}, tokens: ${user.tokenBalance?.balance})`);
}

main()
  .catch((e) => { console.error(e); process.exit(1); })
  .finally(() => prisma.$disconnect());
