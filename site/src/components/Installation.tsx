import { Code, BookOpen, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const Installation = () => {
  return (
    <section className="py-24 px-4 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-background via-card/30 to-background" />
      
      <div className="container max-w-5xl mx-auto relative z-10">
        <div className="text-center mb-12 space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold">
            Get Started in{" "}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Under a Minute
            </span>
          </h2>
          <p className="text-xl text-muted-foreground">
            Install PyTorchLabFlow and start organizing your experiments immediately
          </p>
        </div>

        {/* Installation */}
        <Card className="border-primary/30 bg-card/80 backdrop-blur-sm mb-8">
          <CardContent className="p-8 space-y-6">
            <div className="flex items-center gap-3 mb-4">
              <Code className="h-6 w-6 text-primary" />
              <h3 className="text-2xl font-semibold">Installation</h3>
            </div>
            
            <div className="bg-background/80 p-4 rounded-lg border border-primary/20 font-mono text-sm">
              <code className="text-accent">pip install PyTorchLabFlow</code>
            </div>
          </CardContent>
        </Card>

        {/* Quick Example */}
        <Card className="border-primary/30 bg-card/80 backdrop-blur-sm mb-8">
          <CardContent className="p-8 space-y-6">
            <div className="flex items-center gap-3 mb-4">
              <BookOpen className="h-6 w-6 text-primary" />
              <h3 className="text-2xl font-semibold">Quick Example</h3>
            </div>
            
            <div className="bg-background/80 p-6 rounded-lg border border-primary/20 font-mono text-sm space-y-2 overflow-x-auto">
              <div className="text-muted-foreground"># Setup your project</div>
              <div><span className="text-primary">from</span> PyTorchLabFlow <span className="text-primary">import</span> setup_project</div>
              <div>setup_project(<span className="text-accent">"YourProject"</span>)</div>
              
              <div className="h-4" />
              
              <div className="text-muted-foreground"># Start training</div>
              <div><span className="text-primary">from</span> PyTorchLabFlow <span className="text-primary">import</span> train_new</div>
              <div>train_new(</div>
              <div className="pl-4">name=<span className="text-accent">"exp01"</span>,</div>
              <div className="pl-4">model_loc=<span className="text-accent">"Libs.models.YourModel"</span>,</div>
              <div className="pl-4">dataset_loc=<span className="text-accent">"Libs.datasets.YourDataset"</span></div>
              <div>)</div>
              
              <div className="h-4" />
              
              <div className="text-muted-foreground"># Visualize results</div>
              <div><span className="text-primary">from</span> PyTorchLabFlow <span className="text-primary">import</span> performance_plot</div>
              <div>performance_plot(ppl=<span className="text-accent">"exp01"</span>)</div>
            </div>
          </CardContent>
        </Card>

        {/* CTAs */}
        <div className="flex flex-wrap gap-4 justify-center">
          <Button size="lg" className="gap-2 glow">
            <BookOpen className="h-5 w-5" />
            Read Full Documentation
          </Button>
          <Button size="lg" variant="outline" className="gap-2">
            View Tutorial Notebook
            <ArrowRight className="h-5 w-5" />
          </Button>
        </div>

        {/* Quick links */}
        <div className="mt-12 text-center space-y-4">
          <p className="text-sm text-muted-foreground">
            Want to see it in action? Check out the end-to-end use case:
          </p>
          <a 
            href="https://github.com/BBEK-Anand/Military_AirCraft_Classification" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-primary hover:text-accent transition-colors"
          >
            Military Aircraft Classification Project
            <ArrowRight className="h-4 w-4" />
          </a>
        </div>
      </div>
    </section>
  );
};

export default Installation;
