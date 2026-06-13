import { useState, useRef } from "react";
import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import ResultCard from "@/components/ResultCard";
import FileUploader from "@/components/FileUploader";

const API_URL = "http://127.0.0.1:8000/api/v1/classify";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(API_URL, { method: "POST", body: formData });
      const data = await res.json();
      setResult(data);
    } catch {
      setResult(null);
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  };

  return (
    <div>
      <FileUploader ref={fileInputRef} onFileChange={handleFileChange} />
      <div className="bg-white text-neutral-950 w-full min-h-screen max-w-screen overflow-visible">
        <div className="min-h-956px max-w-1140px flex mx-auto p-8 flex-col gap-8 w-full">
          <Header />
          <main className="grid grid-cols-12 flex-1 gap-8">
            <HeroSection onUpload={handleUploadClick} loading={loading} />
            <aside className="col-span-5 flex flex-col gap-8">
              <ResultCard result={result} />
            </aside>
          </main>

        </div>
      </div>
    </div>
  );
}

export default App;
