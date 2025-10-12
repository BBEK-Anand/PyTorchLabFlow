import { ArrowRight, Github } from "lucide-react";
import { Button } from "@/components/ui/button";
import logo from "@/assets/logo/PTLFlogo-light.jpg";
import chaosVisual from "@/assets/chaos-visual.png";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden py-20 px-4">
      {/* Animated background chaos */}
      <div className="absolute inset-0 opacity-20">
        <img 
          src={chaosVisual} 
          alt="" 
          className="w-full h-full object-cover animate-pulse"
        />
      </div>

      <div className="container max-w-7xl mx-auto relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Logo and Title */}
          <div className="space-y-8 text-center lg:text-left">
            <div className="inline-block">
              <img 
                src={logo} 
                alt="PyTorchLabFlow Logo" 
                className="h-24 w-auto mx-auto lg:mx-0 animate-fade-in"
              />
            </div>
            
            <div className="space-y-4">
              <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
                PyTorch
                <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  LabFlow
                </span>
              </h1>
              
              <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl">
                Research-First Experiment Tracking for Deep Learning Chaos
              </p>
            </div>

            <p className="text-lg text-foreground/80 max-w-2xl">
              Stop wrestling with scattered experiments. Track components, manage artifacts, 
              and organize your research — all locally, no cloud required.
            </p>

            <div className="flex flex-wrap gap-4 justify-center lg:justify-start">
              <Button size="lg" className="gap-2 glow">
                <Github className="h-5 w-5" />
                View on GitHub
              </Button>
              <Button size="lg" variant="outline" className="gap-2">
                Read the Docs
                <ArrowRight className="h-5 w-5" />
              </Button>
            </div>

            <div className="flex flex-wrap gap-6 justify-center lg:justify-start text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-accent animate-pulse" />
                <span>100% Local</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                <span>Zero Config</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-accent animate-pulse" />
                <span>Research-First</span>
              </div>
            </div>
          </div>

          {/* Right: The Mess Visualization */}
          <div className="relative">
            <div className="relative rounded-2xl overflow-hidden border border-primary/20 glow">
              <div className="absolute inset-0 bg-gradient-to-br from-destructive/20 via-primary/10 to-accent/20 animate-pulse" />
              <div className="relative p-8 backdrop-blur-sm">
                <div className="space-y-4">
                  <div className="text-xs text-destructive font-mono opacity-60">
                    // Before PyTorchLabFlow
                  </div>
                  <div className="space-y-2 text-sm font-mono">
                    <div className="text-destructive/80">❌ exp_final_FINAL_v3_actually_final.py</div>
                    <div className="text-destructive/80">❌ Which model was this again?</div>
                    <div className="text-destructive/80">❌ Where did I save those weights?</div>
                    <div className="text-destructive/80">❌ What hyperparameters did I use?</div>
                    <div className="text-destructive/80">❌ 15 untitled notebooks</div>
                  </div>
                  
                  <div className="my-6 text-center">
                    <ArrowRight className="h-8 w-8 mx-auto text-primary animate-pulse" />
                  </div>

                  <div className="text-xs text-accent font-mono opacity-60">
                    // With PyTorchLabFlow
                  </div>
                  <div className="space-y-2 text-sm font-mono">
                    <div className="text-accent">✓ Organized experiments</div>
                    <div className="text-accent">✓ Component tracking</div>
                    <div className="text-accent">✓ Full reproducibility</div>
                    <div className="text-accent">✓ Clear lineage</div>
                    <div className="text-accent">✓ Lab notebook integrated</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
