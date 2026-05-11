"use client"

import * as React from "react"
import { Check, Clipboard } from "lucide-react"

import { Button } from "@/components/ui/button"

type CopyCodeButtonProps = {
  code: string
}

export function CopyCodeButton({ code }: CopyCodeButtonProps) {
  const [copied, setCopied] = React.useState(false)

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      setCopied(false)
    }
  }

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      aria-label="Copy code"
      title="Copy code"
      onClick={copyCode}
      className="absolute right-2 top-2 h-8 w-8 bg-background/80 text-muted-foreground opacity-0 backdrop-blur transition-opacity hover:text-foreground group-hover:opacity-100"
    >
      {copied ? <Check className="h-4 w-4" /> : <Clipboard className="h-4 w-4" />}
    </Button>
  )
}
