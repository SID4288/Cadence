import { Music } from "lucide-react";

function Header() {
  return (
    <header className="shadow-sm rounded-3xl bg-white border-neutral-200 border p-4 flex justify-center items-center">
      <div className="flex items-center gap-3">
        <div className="size-11 rounded-2xl bg-neutral-900 text-neutral-50 flex justify-center items-center">
          <Music className="size-5" />
        </div>
        <div className="flex flex-col">
          <span className="font-semibold text-sm leading-5 tracking-tight">
            Cadence
          </span>
          <span className="text-neutral-500 text-xs leading-4">
            Nepali music genre recognition
          </span>
        </div>
      </div>
    </header>
  );
}

export default Header;
