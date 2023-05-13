##Fakharyar Khan
##Natural Language Processing
##April 1st, 2023
##Professor Sable

                              ##Project 2: CKY PARSER

#parse tree class is used to backtrack through the matrix and retrieve
#a valid parse for a sentence. 

class ParseTree:

  #constructor takes in the root of the tree which for valid parses would
  #be a node holding a constituent S in entry (N, N + 1) in the matrix
  #where N is the number of terms in the sentence
  def __init__(self, root):
    self.root = root


  #this function displays the tree in either the form of a list or as a tree
  #treeMode is a boolean telling us if we should display it as a tree or a list,
  #the level is set by default to 0 and indicates how many nodes away the current node
  #is from the root node. This is only necessary for displaying as a tree
  #Finally the node parameter just holds the current node that we're on and if no value is given
  #we assign it the root node
  
  def display(self, treeMode, level = 0, node = None):
    
    #if node is None then we're getting this function call from the user
    #so we begin displaying the tree from the root.
    if node is None:
      node = self.root

    #the delimiter will be used to separate each of the nodes. If we're printing the tree
    #as an array, we just need to add spaces between each node. But for a tree, for each
    #node we print it on a new line and indent it L times where L is the level of the node

    delimiter = " "
    if treeMode:
      delimiter = "\n" + "\t"*(level+1)
    

    #we add a bracket and display the contents of the node
    printOut = "[" + node.disp

    #then we display the left and right trees
    left = ""
    right = ""
    
    #if the node has a left child
    if node.left != None:
      #we add the delimiter and then display that left branch while also incrementing the level
      left = delimiter + self.display(treeMode, level + 1, node.left)
    
    #and likewise for the right child
    if node.right != None:
      right = delimiter + self.display(treeMode, level + 1, node.right)
    
    #and then we append the left and right branches to printOut
    printOut += left + right
    
    #additionally if we're in treeMode, if a node doesn't have children 
    #we don't add the delimiter and instead just the closing bracket on the same line as the opening brack
    #else after print out the children we add the closing bracket at the same level
    #as the opening bracket

    if treeMode:
      if left == "" and right == "":
        printOut +=  "]"
      else:
        printOut += "\n" + "\t"*level + "]"
    else:
      #and if we're just printing out the array, we can just append the bracket without a delimiter
      printOut += "]"
      
    return printOut

#this is the node class which makes up the tree
#it holds its left and right child, the constituent which is the LHS of the rule
#that the node corresponds to, and disp which holds the constituent and the term it 
#represents if the node corresponds to a terminal

class Node:
  def __init__(self, left, right, constituent, disp):
    self.left = left
    self.right = right
    self.constituent = constituent
    self.disp = disp

#this function reads in the CNF grammar given by the user and 
#fills up a dictionary with those rules
#I decided to have the dictionary map the RHS of the rules to the LHS
#since when filling out the chart, we're trying to find the LHS that consists of
#the RHS

def storeGram(rules):
    #initialize the dictionary
    grammar = {}
    #for each rule in the grammar
    for rule in rules:
      #split the rule by the arrow to get the LHS and the RHS
        component1, component2 = rule.split(" --> ")

        #if this component hasn't been added yet
        if component2 not in grammar:
          #we make an entry in the dictionary and assign it an empty list
          grammar[component2] = []
        
        #then we append component1 to the list of constituents consisting of component2
        grammar[component2].append(component1)
        
    return grammar


#this function implements the CKY algorithm to determine if a sentence is valid as well
#as extract all of the valid parse trees for that sentence
#it takes in the sentence to be parsed and the rule set returned by storeGram
#and returns the filled in parse chart for that sentence

def getChart(sentence, ruleSet):
  
  #we create an (N+1)x(N+1) matrix where N is the length of the sentence
  #whose entries will store a list of nodes
  chart = [ [[] for col in range(len(sentence) + 1)] for row in range(len(sentence) + 1)]

  #for each column (we start from the column 2 since the col # is always greater than the row #)
  for col in range(1, len(sentence) + 1):
    
    #we fill from the bottom up starting from one entry above the main diagonal. 
    #If the terminal spanned by (col -1, col) is in our ruleSet
    if sentence[col - 1] in ruleSet:

      #then we get all of the constituents that consists of that terminal
      #and create a node object for each of them. Since these are terminals
      #they won't have any children so we make those None and we add each of these 
      #nodes to entry (col-1, col) in the chart

      chart[col - 1][col] += [Node(None, None, key, key + " " + sentence[col - 1]) for key in ruleSet[sentence[col - 1]]]

    #after filling that entry, the row # will always be at least 2 fewer than the col #
    #so we fill the matrix from row col - 2 to row 0

    for row in range(col - 2, -1, -1):
      #then for every row < k < col
      for k in range(row + 1, col):
        #we look at the constituents in entries (row, k) and (k, col) called B and C respectively
        for index1 in range(len(chart[row][k])):
          for index2 in range(len(chart[k][col])):
            
            #if the phrase BC is in our rule set
            const = chart[row][k][index1].constituent + " " + chart[k][col][index2].constituent
            if const in ruleSet:
              #we take all of the constituents that contain BC and create a node object for each them
              #Each of these constituents will have their left child be the node representing const.
              #B and the right child, the node representing const. C

              chart[row][col] += [Node(chart[row][k][index1], chart[k][col][index2], key, key) for key in ruleSet[const]]
  return chart

#this is the main driver function that interfaces with the user to get the text file 
#containing the grammar of their language and the sentences that they want parsed

def main():

    #prompt user for text file containing the grammar of their language
    filename = input("Please enter in the name of the text file containing the grammar of your language: ")

    #open the file and split the text by the new line character since each rule is on its own line

    print("Loading grammar...")
    with open(filename, "r") as f:
        ruleSet = storeGram(f.read().splitlines())

        f.close()

    #ask the user if they want textual parse trees to be displayed
    dispTree = input("Do you want textual parse trees to be displayed (y/n)?: ")
    #while the user hasn't quit the program
    while(True):
      #prompt them for a sentence
      sentence = input("Please Enter a Sentence: ")
      
      #if they entered quit, we break out of the loop and terminate the program
      if sentence == "quit":
        print("Goodbye!")
        break;

      #else we call on the getChart function to get the parse matrix for the sentence
      parseMatrix = getChart(sentence.split(), ruleSet)

      #we get all of the nodes in the (0, N) entry in the matrix since 
      #this contains all of the constituents that span the entire sentence.
      #These will be the roots of the parse trees for this sentence

      roots = parseMatrix[0][len(parseMatrix) - 1]

      #we then only take the roots whose constituent is S. If it has any other
      #constituent then it can't be a valid parse of the sentence. We then 
      #construct the parse trees for each of these roots

      trees = [ParseTree(root) for root in roots if root.constituent == "S"]

      #if there were no such roots, we say that there were no valid parses
      if len(trees) == 0:
          print("NO VALID PARSES")
      
      #else we tell the user that it was a valid parse
      else:
          print("VALID SENTENCE\n")
          
          #and display the array representation for each of the parse trees
          for index in range(len(trees)):
            print("Valid parse #" + str(index + 1))
            print(trees[index].display(False))
            

            #and if they wanted the tree representation to be displayed
            if dispTree == "y":
              print("\n")
              #we display that too
              print(trees[index].display(True))
            
            print("\n")

          #and after printing all of the parses, we display the number of valid parses
          #for the sentence
          print("Number of valid parses: " + str(len(trees)) + "\n")
      
main()