import { Sparkles, Upload, FileAudio, BarChart3, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import FeatureCard from "@/components/FeatureCard";

function HeroSection({ onUpload, loading }) {
  return (
    <section className="col-span-7 shadow-sm rounded-3xl bg-white border-neutral-200 border flex p-8 flex-col justify-between">
      <div className="flex flex-col gap-6">
        <div className="inline-flex font-medium rounded-full bg-neutral-100 text-neutral-900 text-xs leading-4 border-neutral-200 border px-4 py-2 items-center gap-2 w-fit">
          <Sparkles className="size-4 text-neutral-900" />
          AI-powered genre detection for Nepali songs
        </div>
        <div className="flex flex-col gap-4">
          <h1 className="max-w-xl font-semibold text-5xl leading-12 tracking-tight">
            Upload a Nepali audio file and instantly discover its genre.
          </h1>
          <p className="max-w-2xl text-neutral-500 text-base leading-7">
            Built for Nepali music only, this platform analyzes your
            track and returns a clear genre prediction with confidence,
            key audio traits, and a simple result summary.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button
            className="font-medium rounded-full bg-neutral-900 text-neutral-50 text-sm leading-5 p-6"
            onClick={onUpload}
            disabled={loading}
          >
            <Upload className="size-4 mr-2" />
            {loading ? "Analyzing..." : "Upload Audio"}
          </Button>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <FeatureCard
          icon={FileAudio}
          title="Upload MP3/WAV"
          description="Drag and drop your Nepali song file for analysis."
        />
        <FeatureCard
          icon={BarChart3}
          title="Genre Prediction"
          description="See the most likely genre with confidence score."
        />
        <FeatureCard
          icon={ShieldCheck}
          title="Nepali Focused"
          description="Optimized for Nepali songs and local music styles."
        />
      </div>
    </section>
  );
}

export default HeroSection;
