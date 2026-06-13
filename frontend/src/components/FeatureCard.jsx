import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

function FeatureCard({ icon: Icon, title, description }) {
  return (
    <Card className="shadow-none rounded-2xl border-0 p-4 gap-4">
      <CardHeader className="p-0 gap-2">
        <div className="size-10 rounded-xl bg-neutral-100 text-neutral-900 flex justify-center items-center">
          <Icon className="size-5" />
        </div>
        <CardTitle className="text-sm leading-5">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-neutral-500 text-sm leading-5 p-0">
        {description}
      </CardContent>
    </Card>
  );
}

export default FeatureCard;
