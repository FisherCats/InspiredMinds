import Link from "next/link";
import { NavDesktopMenu } from "./nav-desktop-menu";
import { NavMobileMenu } from "./nav-mobile-menu";
import { SquareTerminal } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

export function Header() {
  return (
    <header className="pt-4">
      <div
        className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between px-4 sm:px-6"
      >
        {/* Mobile navigation */}
        <NavMobileMenu />

        {/* Logo */}
        <Link href="/" title="Home" className="flex items-center gap-4 md:order-first">
          <SquareTerminal className="w-10 h-10" />
        </Link>

        {/* Desktop navigation */}
        <div className="hidden md:block">
          <NavDesktopMenu />
        </div>

        <ThemeToggle />

        {/* Right side buttons */}
        {/* <div className="flex items-center space-x-2 md:space-x-8 mr-4">
          <Link href="https://github.com/FisherCats" title="Github">
            <GithubIcon />
          </Link>
          <Link href="https://x.com/iguangzhengli" title="X">
            <XIcon />
          </Link>
          <Link href="https://www.xiaohongshu.com/user/profile/6076c9a2000000000101e862" title="Xiaohongshu">
            <XiaohongshuIcon />
          </Link>
        </div> */}
      </div>
    </header >
  );
}
