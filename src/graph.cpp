#include "graph.h"

// Instantiate the graph
KANGraph myGraph;

int main() {
   // Initialize the graph
   myGraph.init();

   // Run the graph for a specified number of iterations
   myGraph.run(100);

   // End the graph execution
   myGraph.end();

   return 0;
}
