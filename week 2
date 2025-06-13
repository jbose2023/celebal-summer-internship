class Node:
    """Represents a node in a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Singly Linked List with basic operations."""
    def __init__(self):
        self.head = None

    def add_node(self, data):
        """Add node with data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        # Traverse to the end
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        """Print all nodes in the list."""
        if not self.head:
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        """Delete the nth node (1-based index)."""
        if not self.head:
            raise Exception("Cannot delete from an empty list.")

        if n <= 0:
            raise Exception("Index must be a positive integer.")

        if n == 1:
            self.head = self.head.next
            return

        current = self.head
        count = 1

        # Traverse to the node before the one we want to delete
        while current and count < n - 1:
            current = current.next
            count += 1

        if not current or not current.next:
            raise Exception(f"Index {n} is out of range.")

        # Bypass the node to delete
        current.next = current.next.next


# --------- Test Code ---------

if __name__ == "__main__":
    ll = LinkedList()

    # Add sample nodes
    ll.add_node(10)
    ll.add_node(20)
    ll.add_node(30)
    ll.add_node(40)
    ll.add_node(50)

    print("Initial List:")
    ll.print_list()

    try:
        print("\nDeleting 3rd node:")
        ll.delete_nth_node(3)
        ll.print_list()

        print("\nDeleting 1st node:")
        ll.delete_nth_node(1)
        ll.print_list()

        print("\nDeleting 10th node (out of range):")
        ll.delete_nth_node(10)
    except Exception as e:
        print(f"Error: {e}")

    try:
        print("\nDeleting from an empty list:")
        empty_list = LinkedList()
        empty_list.delete_nth_node(1)
    except Exception as e:
        print(f"Error: {e}")
