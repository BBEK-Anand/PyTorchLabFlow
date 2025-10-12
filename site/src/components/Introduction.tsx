import { Database, Shield, Zap } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const Introduction = () => {
  const features = [
    {
      icon: Shield,
      title: "100% Offline & Private",
      description: "Run experiments securely on your local machine. No data sharing with third parties. Your research stays yours."
    },
    {
      icon: Zap,
      title: "Zero Reconfiguration",
      description: "Need more power? Transfer your setup to a high-end system without any changes. From laptop to workstation seamlessly."
    },
    {
      icon: Database,
      title: "Research-First Design",
      description: "Built by a researcher, for researchers. Track components, organize by hypothesis, and maintain a digital lab notebook."
    }
  ];

  return (
    <section className="py-24 px-4">
      <div className="container max-w-7xl mx-auto">
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold">
            Your Go-To Solution for{" "}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              PyTorch Experiments
            </span>
          </h2>
          <p className="text-xl text-muted-foreground">
            PyTorchLabFlow is your offline companion for managing deep learning experiments with ease. 
            Track every component of your training pipeline, maintain reproducibility, and never lose context again.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Card 
              key={index} 
              className="border-primary/20 bg-card/50 backdrop-blur-sm hover:border-primary/40 transition-all hover:glow"
            >
              <CardContent className="p-6 space-y-4">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Introduction;
