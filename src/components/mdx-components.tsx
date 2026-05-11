import * as React from "react"
import Image from "next/image"
import Link from "next/link"

import { cn } from "@/lib/utils"

const components = {
  h1: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1
      className={cn(
        "font-heading mt-12 mb-7 scroll-m-20 text-[2.35rem] leading-tight font-semibold tracking-tight sm:text-[3.15rem]",
        className
      )}
      {...props}
    />
  ),
  h2: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2
      className={cn(
        "font-heading mt-11 mb-5 scroll-m-20 text-[1.7rem] leading-[2.25rem] font-semibold tracking-tight sm:text-[1.95rem]",
        className
      )}
      {...props}
    />
  ),
  h3: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h3
      className={cn(
        "font-heading mt-9 mb-4 scroll-m-20 text-[1.4rem] leading-[2rem] font-semibold tracking-tight sm:text-[1.58rem]",
        className
      )}
      {...props}
    />
  ),
  h4: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h4
      className={cn(
        "font-heading mt-8 mb-4 scroll-m-20 text-[1.25rem] leading-[1.9rem] font-semibold tracking-tight",
        className
      )}
      {...props}
    />
  ),
  h5: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h5
      className={cn(
        "font-heading mt-6 mb-3 scroll-m-20 text-[1.1rem] leading-[1.7rem] font-semibold tracking-tight",
        className
      )}
      {...props}
    />
  ),
  h6: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h6
      className={cn(
        "font-heading mt-6 mb-3 scroll-m-20 text-[1rem] leading-[1.6rem] font-semibold tracking-tight",
        className
      )}
      {...props}
    />
  ),
  a: ({ className, ...props }: React.HTMLAttributes<HTMLAnchorElement>) => (
    <a
      className={cn(
        "font-medium text-foreground transition-all duration-300 ease-in-out underline underline-offset-4 break-words",
        className
      )}
      {...props}
    />
  ),
  p: ({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p
      className={cn(
        "my-5 text-[1.08rem] leading-[2rem] sm:text-[1.16rem] sm:leading-[2.15rem]",
        className
      )}
      {...props}
    />
  ),
  strong: ({ className, ...props }: React.HTMLAttributes<HTMLElement>) => (
    <strong className={cn("font-bold", className)} {...props} />
  ),
  ul: ({ className, ...props }: React.HTMLAttributes<HTMLUListElement>) => (
    <ul className={cn("my-6 ml-5 list-disc space-y-2 sm:ml-6", className)} {...props} />
  ),
  ol: ({ className, ...props }: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className={cn("my-6 ml-5 list-decimal space-y-2 sm:ml-6", className)} {...props} />
  ),
  li: ({ className, ...props }: React.HTMLAttributes<HTMLElement>) => (
    <li className={cn("pl-1 text-[1.08rem] leading-[1.9rem] sm:text-[1.16rem]", className)} {...props} />
  ),
  blockquote: ({ className, ...props }: React.HTMLAttributes<HTMLElement>) => (
    <blockquote
      className={cn(
        "my-6 border-l-[3px] border-border pl-4",
        className
      )}
      {...props}
    />
  ),
  img: ({
    className,
    alt,
    ...props
  }: React.ImgHTMLAttributes<HTMLImageElement>) => (
    // eslint-disable-next-line @next/next/no-img-element
    <img className={cn("mx-auto my-6 h-auto max-w-full rounded-md object-contain", className)} alt={alt} {...props} />
  ),
  hr: ({ ...props }: React.HTMLAttributes<HTMLHRElement>) => (
    <hr className="my-4 md:my-8" {...props} />
  ),
  table: ({ className, ...props }: React.HTMLAttributes<HTMLTableElement>) => (
    <div className="my-6 w-full overflow-x-auto">
      <table
        className={cn(
          "mx-auto mb-12 w-full min-w-max border-collapse border-spacing-0 text-left",
          "text-[1rem] sm:text-[1.1rem]",
          className
        )}
        {...props}
      />
    </div>
  ),
  tr: ({ className, ...props }: React.HTMLAttributes<HTMLTableRowElement>) => (
    <tr
      className={cn("m-0", className)}
      {...props}
    />
  ),
  th: ({ className, ...props }: React.HTMLAttributes<HTMLTableCellElement>) => (
    <th
      className={cn(
        "border-b border-solid border-border p-4 text-left font-semibold",
        className
      )}
      {...props}
    />
  ),
  td: ({ className, ...props }: React.HTMLAttributes<HTMLTableCellElement>) => (
    <td
      className={cn(
        "border-b border-dashed border-border p-4 text-left leading-[1.5]",
        className
      )}
      {...props}
    />
  ),
  pre: ({ className, ...props }: React.HTMLAttributes<HTMLPreElement>) => (
    <pre
      className={cn(
        "my-7 overflow-x-auto rounded-md bg-[#0d1117] p-5 text-[0.95rem] leading-7",
        className
      )}
      {...props}
    />
  ),
  code: ({ className, ...props }: React.HTMLAttributes<HTMLElement>) => {
    const isBlockCode = className?.includes("language-") || className?.includes("hljs")

    return (
      <code
        className={cn(
          isBlockCode
            ? "block min-w-max bg-transparent p-0 font-mono text-[0.95rem] font-normal leading-7 text-inherit"
            : "rounded-md bg-muted px-[0.3rem] py-[0.15rem] font-mono text-[0.95em] font-normal text-[#c92a3d] break-words dark:text-[#ff7b8a]",
          className
        )}
        {...props}
      />
    )
  },
  small: ({ className, ...props }: React.HTMLAttributes<HTMLElement>) => (
    <small
      className={cn("text-[70%]", className)}
      {...props}
    />
  ),
  Image,
  Link: ({ className, ...props }: React.ComponentProps<typeof Link>) => (
    <Link
      className={cn(
        "font-medium text-blue-600 no-underline transition-all duration-300 ease-in-out hover:underline",
        className
      )}
      {...props}
    />
  ),
  LinkedCard: ({ className, ...props }: React.ComponentProps<typeof Link>) => (
    <Link
      className={cn(
        "flex w-full flex-col items-center rounded-xl border bg-card p-6 text-card-foreground shadow transition-colors hover:bg-muted/50 sm:p-10",
        className
      )}
      {...props}
    />
  ),
}

export { components }
