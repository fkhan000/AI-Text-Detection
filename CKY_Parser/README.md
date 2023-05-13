##Instructions on Running the Program

To run the program you need python 3.9.12 or a later version installed. The program doesn’t require any external libraries.

The program will prompt you to enter the path to the text file that contains the CNF converted rules of the grammar you want
to use. After it finishes loading in the rule set, it will ask if you want the parse trees to be displayed to which you 
should reply with either y for yes or n for no. Finally, it will prompt you for the sentence that you want to parse. 
Once it receives it, it will tell you whether or not the sentence is valid according to the grammar you have given it 
and if it is, it will print out all of the valid parse trees for that sentence and will then ask for another sentence 
to be parsed. If you ever want to quit the program enter the word quit in all lowercase.

##Implementation Details

The first decision I needed to make in implementing the parser was how I wanted to represent the grammar that the user gives.
At first, I thought that I could store the grammar in a dictionary mapping the LHS of each rule to its RHS. However, this
ended up being inefficient as when we’re filling out the chart in the CKY algorithm, we want to find the constituent that
consists of a non-terminal or two terminals. So it would be much more natural and efficient to have map the RHS of each rule
to its LHS. I think that with the way that the rule was formatted, LHS -> RHS, made it seem intuitive to me that we needed to
have the dictionary go from LHS to RHS when in reality, RHS -> LHS fits much more well with the CKY algorithm. For implementing
the CKY algorithm, I decided to have each entry in the chart contain a list of nodes. Each node holds a left and right child and 
the LHS constituent that it represents. If the node corresponds to a terminal, it has no children so the left and right child are None.
Else the left and right child are the nodes that correspond to the two constituents that the LHS constituent consists of. Once the chart
is filled up, the program takes all of the nodes in entry (0, N) of the matrix whose constituent is “S”. These are the root nodes of all
of the valid parse trees for the sentence and we can use this node to print out each parse tree.

I technically could have made the function that prints out the tree a class method for the node class. However, I decided to instead 
make a separate class called parseTree and placed the function there. I don’t know if this was the best decision because the constructor
for that class just passes in the root node for the tree so it seems really unnecessary. But it also seems weird to me to have a method
for printing out the tree in a node class. I guess you could argue that each node is printing out the subtree underneath it but if I
ever needed to display just the node I would need two different display functions for one class. I looked up online which
way was better and it looks like it can be implemented either way. I found a good argument for making a separate class though: with just
the node class, an empty tree would just be None which isn’t correct whereas with a separate class, the root node would be None but the
tree object wouldn’t be. That situation doesn’t really come up in this project but I think it’s good practice.
