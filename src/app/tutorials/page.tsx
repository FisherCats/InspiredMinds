import type { Metadata } from "next"
import Link from "next/link"
import count from "word-count"
import { allTutorials } from "content-collections"

import { config } from "@/lib/config"

export const metadata: Metadata = {
  title: `Tutorials | ${config.site.title}`,
  description: `Learning notes and tutorials from ${config.site.title}`,
}

type TutorialsPageProps = {
  searchParams: Promise<{ tag?: string }>
}

function formatDate(date: string) {
  const value = new Date(date)
  return `${value.getFullYear()}年${value.getMonth() + 1}月${value.getDate()}日`
}

export default async function TutorialsPage({ searchParams }: TutorialsPageProps) {
  const { tag: selectedTag } = await searchParams
  const tutorials = [...allTutorials].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  )
  const tags = Array.from(
    new Set(tutorials.flatMap((tutorial) => tutorial.tags ?? []))
  ).sort()
  const filteredTutorials = selectedTag
    ? tutorials.filter((tutorial) => tutorial.tags?.includes(selectedTag))
    : tutorials

  return (
    <main className="mx-auto w-full max-w-4xl px-4 py-8 sm:px-6">
      <div className="mb-10">
        <h1 className="text-3xl font-bold">Tutorials</h1>
        <p className="mt-2 text-muted-foreground">学习记录与实践笔记</p>
      </div>

      <div className="mb-10 flex flex-wrap gap-2" aria-label="按标签筛选">
        <Link
          href="/tutorials"
          className={`rounded-md border px-3 py-1 text-sm transition-colors ${
            selectedTag
              ? "border-border text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              : "border-foreground bg-foreground text-background"
          }`}
        >
          All
        </Link>
        {tags.map((tag) => (
          <Link
            key={tag}
            href={`/tutorials?tag=${encodeURIComponent(tag)}`}
            className={`rounded-md border px-3 py-1 text-sm transition-colors ${
              selectedTag === tag
                ? "border-foreground bg-foreground text-background"
                : "border-border text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            }`}
          >
            {tag}
          </Link>
        ))}
      </div>

      <div className="space-y-8">
        {filteredTutorials.map((tutorial) => (
          <article key={tutorial.slug}>
            <Link href={`/tutorials/${tutorial.slug}`}>
              <div className="flex flex-col space-y-2">
                <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                  <h2 className="text-xl font-semibold underline underline-offset-4">
                    {tutorial.title}
                  </h2>
                  <span className="text-sm text-muted-foreground">
                    {formatDate(tutorial.date)} · {count(tutorial.content)} 字
                  </span>
                </div>
                {tutorial.summary ? (
                  <p className="line-clamp-2 text-muted-foreground">{tutorial.summary}</p>
                ) : null}
                {tutorial.tags.length > 0 ? (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {tutorial.tags.map((tag) => (
                      <span key={tag} className="text-xs text-muted-foreground">
                        #{tag}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            </Link>
          </article>
        ))}
      </div>
    </main>
  )
}
