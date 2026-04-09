import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function displayName(p: {
  alias?: string | null;
  title?: string | null;
  name: string;
}): string {
  return (
    p.alias || p.title || p.name.replace(/_/g, " ").replace(/-/g, " ")
  );
}
