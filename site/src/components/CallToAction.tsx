import { Users, Github, HeartHandshake, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const CallToAction = () => {
  return (
    <section className="py-24 px-4">
      <div className="container max-w-7xl mx-auto space-y-16">
        {/* Collaboration Invite */}
        <div className="text-center max-w-3xl mx-auto space-y-6">
          <h2 className="text-4xl md:text-5xl font-bold">
            Let's Build{" "}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Together
            </span>
          </h2>
          <p className="text-xl text-muted-foreground">
            PyTorchLabFlow is early-stage and evolving fast. If you're doing deep learning research 
            and facing similar frustrations, let's collaborate.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* For Researchers */}
          <Card className="border-primary/30 bg-gradient-to-br from-card/80 to-primary/5 backdrop-blur-sm">
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center mb-4">
                <Users className="h-6 w-6 text-primary" />
              </div>
              <CardTitle className="text-2xl">For Researchers</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                Working on messy, exciting deep learning projects? Tired of disorganized experiments 
                and tool overload? Let's shape PyTorchLabFlow together based on your real workflows.
              </p>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-accent">✓</span>
                  <span>Test features early and provide feedback</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-accent">✓</span>
                  <span>Shape the tool to fit your research needs</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-accent">✓</span>
                  <span>Get hands-on support from the creator</span>
                </li>
              </ul>
              <Button className="w-full gap-2">
                <HeartHandshake className="h-5 w-5" />
                Join as Research Collaborator
              </Button>
            </CardContent>
          </Card>

          {/* For Contributors */}
          <Card className="border-accent/30 bg-gradient-to-br from-card/80 to-accent/5 backdrop-blur-sm">
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-accent/20 to-primary/20 flex items-center justify-center mb-4">
                <Github className="h-6 w-6 text-accent" />
              </div>
              <CardTitle className="text-2xl">For Developers</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                Help evolve PyTorchLabFlow into the definitive research experiment tracker. 
                Fork the repo, open issues, submit PRs — let's build something useful together.
              </p>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-primary">✓</span>
                  <span>Contribute to a tool researchers actually need</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">✓</span>
                  <span>Work on interesting ML infrastructure problems</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">✓</span>
                  <span>Join an early-stage open source project</span>
                </li>
              </ul>
              <div className="flex gap-3">
                <Button className="flex-1 gap-2">
                  <Github className="h-5 w-5" />
                  Fork on GitHub
                </Button>
                <Button variant="outline" className="flex-1 gap-2">
                  <Star className="h-5 w-5" />
                  Star Repo
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Repository stats */}
        <Card className="border-primary/20 bg-card/50 backdrop-blur-sm max-w-3xl mx-auto">
          <CardContent className="p-8">
            <div className="flex flex-wrap justify-around gap-8 text-center">
              <div>
                <div className="text-3xl font-bold text-primary mb-1">5</div>
                <div className="text-sm text-muted-foreground">GitHub Stars</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-accent mb-1">1</div>
                <div className="text-sm text-muted-foreground">Forks</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-primary mb-1">Apache 2.0</div>
                <div className="text-sm text-muted-foreground">License</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-accent mb-1">Active</div>
                <div className="text-sm text-muted-foreground">Development</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default CallToAction;
