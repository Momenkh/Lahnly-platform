export interface AuthUser {
  id: string;
  clerkId: string;
  email: string;
  displayName: string | null;
  createdAt: Date;
}
