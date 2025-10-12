import { Layers, Box, GitMerge, FlaskConical } from "lucide-react";
import organizedFlow from "@/assets/organized-flow.png";

const HowItWorks = () => {
  const features = [
    {
      icon: Layers,
      title: "Component Tracking",
      description: "Track individual layers, attention blocks, normalizations — not just whole models. See how each component performs across different architectures and understand what really matters."
    },
    {
      icon: Box,
      title: "Flexible Nesting",
      description: "Organize your work how you think. Nest experiments, group by hypothesis, create hierarchies that match your research workflow. No more flat lists of disconnected runs."
    },
    {
      icon: GitMerge,
      title: "Pipeline Class",
      description: "The Pipeline class manages your entire experiment lifecycle — from setup through training to analysis. Configure once, reuse everywhere. Change one component, see the impact across all runs."
    },
    {
      icon: FlaskConical,
      title: "Digital Lab Notebook",
      description: "Every observation, insight, and hypothesis gets tracked alongside your metrics. Future you will thank past you for writing down why Run 42 had that weird behavior."
    }
  ];

  return (
    <section className="py-24 px-4">
      <div className="container max-w-7xl mx-auto">
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold">
            How PyTorchLabFlow{" "}
            <span className="bg-gradient-to-r from-accent to-primary bg-clip-text text-transparent">
              Manages Chaos
            </span>
          </h2>
          <p className="text-xl text-muted-foreground">
            Built from the ground up with research workflows in mind. 
            Everything you need to go from chaos to clarity.
          </p>
        </div>

        {/* Visual workflow */}
        <div className="mb-16 rounded-2xl overflow-hidden border border-primary/20 glow">
          <img 
            src={organizedFlow} 
            alt="Organized workflow visualization" 
            className="w-full h-auto"
          />
        </div>

        {/* Features grid */}
        <div className="grid md:grid-cols-2 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center flex-shrink-0">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-2xl font-semibold">{feature.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Workflow summary */}
        <div className="mt-16 p-8 rounded-2xl bg-gradient-to-br from-card/50 to-primary/5 border border-primary/20">
          <h3 className="text-2xl font-semibold mb-6 text-center">Your Research Workflow</h3>
          <div className="flex flex-wrap justify-center gap-4 text-center">
            <div className="flex items-center gap-3 px-6 py-3 rounded-lg bg-background/50 backdrop-blur-sm border border-primary/20">
              <span className="text-2xl font-bold text-primary">1</span>
              <span className="text-sm">Setup Project</span>
            </div>
            <div className="text-2xl text-muted-foreground">→</div>
            <div className="flex items-center gap-3 px-6 py-3 rounded-lg bg-background/50 backdrop-blur-sm border border-primary/20">
              <span className="text-2xl font-bold text-primary">2</span>
              <span className="text-sm">Organize Components</span>
            </div>
            <div className="text-2xl text-muted-foreground">→</div>
            <div className="flex items-center gap-3 px-6 py-3 rounded-lg bg-background/50 backdrop-blur-sm border border-primary/20">
              <span className="text-2xl font-bold text-primary">3</span>
              <span className="text-sm">Configure & Train</span>
            </div>
            <div className="text-2xl text-muted-foreground">→</div>
            <div className="flex items-center gap-3 px-6 py-3 rounded-lg bg-background/50 backdrop-blur-sm border border-primary/20">
              <span className="text-2xl font-bold text-accent">4</span>
              <span className="text-sm">Analyze & Decide</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
