import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fuzzyMatch(text: string, query: string): number {
  let qi = 0;
  let score = 0;
  let lastMatch = -1;
  const tl = text.toLowerCase();
  const ql = query.toLowerCase();
  for (let ti = 0; ti < tl.length && qi < ql.length; ti++) {
    if (tl[ti] === ql[qi]) {
      score += ti === lastMatch + 1 ? 2 : 1;
      if (
        ti === 0 ||
        tl[ti - 1] === " " ||
        tl[ti - 1] === "-" ||
        tl[ti - 1] === "_"
      )
        score += 3;
      lastMatch = ti;
      qi++;
    }
  }
  return qi === ql.length ? score : 0;
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
