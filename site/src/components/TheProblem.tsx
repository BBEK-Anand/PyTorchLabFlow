import { AlertCircle, FileQuestion, GitBranch, Lightbulb, RotateCcw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const TheProblem = () => {
  const problems = [
    {
      icon: GitBranch,
      title: "Can't Track Components",
      description: "Testing attention blocks across architectures? Swapping normalizations? Tools treat models as black boxes — they don't understand the research you're actually doing.",
      impact: "Lost insights, repeated work, unclear causality"
    },
    {
      icon: FileQuestion,
      title: "Artifacts Float Without Context",
      description: "5 data splits, 5 results, 5 sets of predictions. But which config? Which seed? Which version? The connections vanish into manual folders and spreadsheets.",
      impact: "Can't reproduce, can't compare, can't trust results"
    },
    {
      icon: Lightbulb,
      title: "Hypotheses Get Lost",
      description: "You're not just running experiments — you're testing ideas. But tools give you flat lists with tags, not trees of thought, not the evolution of your research.",
      impact: "No narrative, no learning, no research story"
    },
    {
      icon: AlertCircle,
      title: "Observations Have Nowhere to Go",
      description: "\"Run 42 had weird variance\" — where does that insight live? Not in the tool. Maybe in a notes field. Maybe lost forever.",
      impact: "Best insights come from failures you forget"
    },
    {
      icon: RotateCcw,
      title: "Reuse Is Pure Pain",
      description: "Built the perfect tokenizer? Great optimizer config? Now copy-paste it 10 times and pray you don't break something. No registry, no tracking, no help.",
      impact: "Reinventing wheels, introducing bugs, wasting time"
    }
  ];

  return (
    <section className="py-24 px-4 relative overflow-hidden">
      {/* Background accent */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-card/50 to-background" />
      
      <div className="container max-w-7xl mx-auto relative z-10">
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold">
            The Real Problem with{" "}
            <span className="bg-gradient-to-r from-destructive to-primary bg-clip-text text-transparent">
              Existing Tools
            </span>
          </h2>
          <p className="text-xl text-muted-foreground">
            Most experiment trackers are built for ML engineering, not discovery-driven research. 
            Here's what they get wrong — and what it costs you.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {problems.map((problem, index) => (
            <Card 
              key={index}
              className="border-destructive/20 bg-card/80 backdrop-blur-sm hover:border-destructive/40 transition-all"
            >
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-destructive/20 to-primary/20 flex items-center justify-center mb-4">
                  <problem.icon className="h-6 w-6 text-destructive" />
                </div>
                <CardTitle className="text-xl">{problem.title}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {problem.description}
                </p>
                <p className="text-xs text-destructive/80 italic border-l-2 border-destructive/40 pl-3">
                  {problem.impact}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center">
          <Card className="border-primary/30 bg-card/50 backdrop-blur-sm max-w-3xl mx-auto">
            <CardContent className="p-8">
              <p className="text-lg leading-relaxed text-foreground/90">
                As an MSc Data Science student, I lived this pain. Every experiment had too many moving parts: 
                layers, hyperparameters, datasets, metrics, losses, random seeds, and logs. I tried MLflow, W&B, DVC… 
                and while they <span className="text-primary font-semibold">help</span>, none felt designed for 
                <span className="text-accent font-semibold"> research the way I wanted to do it</span>.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default TheProblem;
