import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { siteUrl } from "@/lib/config"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function absoluteUrl(path: string) {
  return `${siteUrl}${path}`
}
