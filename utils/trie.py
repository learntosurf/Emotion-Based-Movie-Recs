class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.is_end_of_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                return False
            current = current.children[letter]
        return current.is_end_of_word

    def starts_with(self, prefix):
        current = self.root
        for letter in prefix:
            if letter not in current.children:
                return []
            current = current.children[letter]
        return self._elements_with_prefix(current, prefix)

    def _elements_with_prefix(self, node, prefix):
        elements = []
        if node.is_end_of_word:
            elements.append(prefix)
        for letter, child in node.children.items():
            elements.extend(self._elements_with_prefix(child, prefix + letter))
        return elements
