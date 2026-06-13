import { Headphones } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

function ResultCard({ result }) {
  return (
    <Card className="shadow-sm rounded-3xl border-0 p-6 gap-4">
      <CardHeader className="p-0 gap-2">
        <CardTitle className="text-lg leading-7">
          {result ? "Result" : "Sample result"}
        </CardTitle>
        <CardDescription>
          {result
            ? `Prediction for ${result.filename}`
            : "Upload a file to see the prediction."}
        </CardDescription>
      </CardHeader>
      {result && (
        <CardContent className="p-0 gap-4 flex flex-col">
          <div className="rounded-2xl bg-neutral-100 flex p-4 justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="size-11 rounded-xl bg-neutral-900 text-neutral-50 flex justify-center items-center">
                <Headphones className="size-5" />
              </div>
              <div className="flex flex-col">
                <span className="font-medium text-sm leading-5">
                  Detected Genre
                </span>
                <span className="text-neutral-500 text-xs leading-4">
                  Top prediction
                </span>
              </div>
            </div>
            <Badge className="rounded-full bg-neutral-900 text-neutral-50 px-3 py-1 capitalize">
              {result.predicted_genre.replace(/_/g, " ")}
            </Badge>
          </div>
          <div className="rounded-2xl border border-neutral-200 p-4">
            <p className="uppercase text-neutral-500 text-xs leading-4 tracking-wide">
              Confidence
            </p>
            <p className="font-semibold text-2xl leading-8 mt-2">
              {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </CardContent>
      )}
    </Card>
  );
}

export default ResultCard;
