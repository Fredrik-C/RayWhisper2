# Domain Driven Design (DDD)

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Building Blocks](#building-blocks)
4. [Strategic Design](#strategic-design)
5. [Tactical Design](#tactical-design)
6. [Implementation Patterns](#implementation-patterns)
7. [Common Pitfalls](#common-pitfalls)
8. [Best Practices](#best-practices)
9. [References](#references)

## Introduction

### What is Domain Driven Design?

Domain Driven Design (DDD) is a software development approach that places the business domain at the center of the architecture and design process. Coined by Eric Evans in his 2003 book "Domain-Driven Design: Tackling Complexity in the Heart of Software," DDD provides a framework for handling complex business logic by aligning the code structure with the business domain.

### Why DDD Matters

- **Complexity Management**: Breaks down complex business problems into manageable pieces
- **Communication Bridge**: Creates a common language between developers and domain experts
- **Maintainability**: Results in code that is easier to understand and modify
- **Scalability**: Enables systems to grow and evolve with changing business needs
- **Quality**: Focuses on modeling the domain correctly, leading to fewer bugs and better design

### The Core Philosophy

> "The heart of software is its ability to solve domain-related problems for its user. All other features, functions, and characteristics serve this basic purpose."

DDD emphasizes:
- Understanding the business domain deeply
- Creating a shared language (Ubiquitous Language) between technical and non-technical stakeholders
- Organizing code around domain concepts rather than technical layers
- Making business logic explicit and testable

## Core Concepts

### Ubiquitous Language

The Ubiquitous Language is a shared vocabulary developed collaboratively by developers and domain experts. It serves as the cornerstone of DDD.

**Characteristics:**
- Explicitly defined terms for domain concepts
- Used consistently in conversations, documentation, and code
- Evolves as understanding deepens
- Reduces miscommunication and ambiguity

**Example:**
Instead of database-centric terms like "user_record" or "transaction_row," use domain terms like "Customer," "Order," or "Payment."

### Domain

The domain is the subject area upon which the software is focused. Understanding the domain is the first priority in DDD.

**Types of Domains:**
- **Core Domain**: The part that differentiates your business (primary focus)
- **Supporting Domain**: Necessary but not differentiating (e.g., billing, reporting)
- **Generic Domain**: Standardized solutions that exist off-the-shelf (e.g., authentication, logging)

### Subdomains

Complex domains are subdivided into subdomains, each representing a specific area of the business.

**Benefits:**
- Divide and conquer complexity
- Allow teams to focus on specific areas
- Enable parallel development
- Reduce cognitive load

**Types of Subdomains:**
- **Core Subdomains**: Critical to business success
- **Supporting Subdomains**: Necessary but not differentiating
- **Generic Subdomains**: Can be outsourced or bought

### Bounded Context

A bounded context is an explicit boundary within which a domain model is valid. Different bounded contexts can have different models for the same concept.

**Key Principles:**
- Each bounded context has its own ubiquitous language
- Models are isolated within their boundaries
- Communication between contexts is explicit and controlled
- Prevents model pollution and conflicting requirements

**Example:**
In an e-commerce system:
- **Sales Context**: Product means something available for purchase
- **Inventory Context**: Product means something in stock
- **Shipping Context**: Product means something that needs to be packaged

## Building Blocks

### Entities

**Definition**: Objects that have a distinct identity and lifecycle.

**Characteristics:**
- Continuity and identity matter more than attributes
- Can change attributes while maintaining identity
- Have a lifecycle (creation, modification, deletion)
- Mutable

**Example:**
```python
class Customer:
    def __init__(self, customer_id: str, name: str, email: str):
        self.customer_id = customer_id  # Identity
        self.name = name
        self.email = email
    
    def update_email(self, new_email: str):
        self.email = new_email
    
    def __eq__(self, other):
        return isinstance(other, Customer) and self.customer_id == other.customer_id
```

### Value Objects

**Definition**: Objects that describe a thing without having an identity.

**Characteristics:**
- Identity is not important; attributes define them
- Immutable (once created, cannot be changed)
- Equality based on all attributes, not identity
- Can be shared freely
- Interchangeable if attributes are identical

**Example:**
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Money:
    amount: float
    currency: str
    
    def add(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)

@dataclass(frozen=True)
class Address:
    street: str
    city: str
    postal_code: str
    country: str
```

### Aggregates

**Definition**: A cluster of domain objects bound together by a root entity (Aggregate Root).

**Purpose:**
- Ensure consistency within a boundary
- Simplify relationships between entities
- Define transaction boundaries
- Protect invariants

**Rules:**
- Access other aggregates only through their root
- Delete entire aggregate or nothing
- Reference other aggregates by ID only, not by object reference
- Maintain internal consistency

**Example:**
```python
class Order:  # Aggregate Root
    def __init__(self, order_id: str, customer_id: str):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items: List[OrderItem] = []
        self.status = OrderStatus.PENDING
    
    def add_item(self, product_id: str, quantity: int, price: Money):
        # Business logic to maintain invariants
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        item = OrderItem(product_id, quantity, price)
        self.items.append(item)
    
    def place_order(self):
        if not self.items:
            raise ValueError("Order must have at least one item")
        self.status = OrderStatus.PLACED

class OrderItem:  # Part of Order aggregate
    def __init__(self, product_id: str, quantity: int, price: Money):
        self.product_id = product_id
        self.quantity = quantity
        self.price = price
```

### Repositories

**Definition**: Mediates between the domain and data mapping layers, acting like an in-memory collection.

**Purpose:**
- Abstract persistence concerns from domain logic
- Provide a domain-oriented interface for accessing aggregates
- Hide database implementation details
- Enable testing with mock repositories

**Example:**
```python
from abc import ABC, abstractmethod

class OrderRepository(ABC):
    @abstractmethod
    def save(self, order: Order) -> None:
        pass
    
    @abstractmethod
    def find_by_id(self, order_id: str) -> Order:
        pass
    
    @abstractmethod
    def find_by_customer(self, customer_id: str) -> List[Order]:
        pass

class DatabaseOrderRepository(OrderRepository):
    def __init__(self, db_connection):
        self.db = db_connection
    
    def save(self, order: Order) -> None:
        # Persist to database
        pass
    
    def find_by_id(self, order_id: str) -> Order:
        # Retrieve from database
        pass
```

### Services

**Definition**: Objects that perform an action or operation that doesn't naturally belong to an entity or value object.

**Characteristics:**
- Stateless
- Named after domain actions (verbs)
- Depend on domain objects
- Should be minimal and focused

**Example:**
```python
class OrderService:
    def __init__(self, order_repo: OrderRepository, payment_service: PaymentService):
        self.order_repo = order_repo
        self.payment_service = payment_service
    
    def checkout(self, order_id: str, payment_details: PaymentDetails) -> bool:
        order = self.order_repo.find_by_id(order_id)
        total = self._calculate_total(order)
        
        if self.payment_service.process(payment_details, total):
            order.place_order()
            self.order_repo.save(order)
            return True
        return False
```

### Factories

**Definition**: Objects responsible for creating complex aggregates or entities.

**Purpose:**
- Encapsulate complex creation logic
- Ensure valid object creation
- Hide internal structure

**Example:**
```python
class OrderFactory:
    @staticmethod
    def create_order(customer_id: str, items: List[dict]) -> Order:
        order = Order(str(uuid.uuid4()), customer_id)
        
        for item in items:
            product = get_product(item['product_id'])  # Domain logic
            order.add_item(
                product.product_id,
                item['quantity'],
                product.price
            )
        
        return order
```

### Domain Events

**Definition**: Something that happened in the domain that domain experts care about.

**Purpose:**
- Capture state changes
- Enable communication between aggregates and bounded contexts
- Create audit trails
- Support eventual consistency

**Example:**
```python
@dataclass
class OrderPlaced:
    order_id: str
    customer_id: str
    total_amount: Money
    timestamp: datetime
    
    def handle(self):
        # Trigger side effects like sending confirmation email
        pass
```

## Strategic Design

### Subdomains and Bounded Contexts

Strategic design deals with the big picture:

**Mapping Subdomains to Bounded Contexts:**
- Each significant subdomain should map to one or more bounded contexts
- Align organizational structure with bounded contexts (Conway's Law)
- Define clear responsibilities for each context

### Context Mapping

Context Mapping defines how bounded contexts relate to and communicate with each other.

**Patterns:**

1. **Partnership**: Two contexts work together closely, coordinating frequently
2. **Shared Kernel**: Two contexts share a subset of the model
3. **Customer-Supplier**: One context (customer) depends on another (supplier)
4. **Conformist**: One context conforms to the model of another
5. **Anti-Corruption Layer**: A context translates another's model to protect its own
6. **Separate Ways**: Contexts are isolated with no interaction

**Anti-Corruption Layer Example:**
```python
# Old Legacy System
class LegacyCustomer:
    def __init__(self, cust_id, full_name, addr_string):
        self.cust_id = cust_id
        self.full_name = full_name
        self.addr_string = addr_string

# Our Domain Model
@dataclass
class Customer:
    customer_id: str
    name: str
    address: Address

# Anti-Corruption Layer (Adapter)
class LegacyCustomerAdapter:
    def __init__(self, legacy_repo):
        self.legacy_repo = legacy_repo
    
    def get_customer(self, customer_id: str) -> Customer:
        legacy_customer = self.legacy_repo.find(customer_id)
        
        # Translate legacy model to our domain model
        street, city, postal = legacy_customer.addr_string.split(',')
        address = Address(street, city, postal, "USA")
        
        return Customer(
            customer_id=legacy_customer.cust_id,
            name=legacy_customer.full_name,
            address=address
        )
```

## Tactical Design

### Layered Architecture

A common architecture for DDD projects:

```
┌─────────────────────────────┐
│   Presentation Layer        │  (UI, API Controllers)
├─────────────────────────────┤
│   Application Layer         │  (Use Cases, Orchestration)
├─────────────────────────────┤
│   Domain Layer              │  (Entities, Value Objects, Services)
├─────────────────────────────┤
│   Infrastructure Layer      │  (Persistence, External Services)
└─────────────────────────────┘
```

**Layer Responsibilities:**

- **Presentation**: Handles user interaction and formatting responses
- **Application**: Orchestrates domain objects to accomplish business tasks
- **Domain**: Contains the heart of the business logic
- **Infrastructure**: Provides technical capabilities

### Event Sourcing

Instead of storing current state, store all state changes as events.

**Benefits:**
- Complete audit trail
- Time travel debugging
- Event replay
- Scalability through eventual consistency

**Example:**
```python
class Order:
    def __init__(self):
        self.uncommitted_events = []
        self.status = None
        self.items = []
    
    def place_order(self):
        event = OrderPlacedEvent(self.order_id, self.customer_id)
        self._apply_event(event)
        self.uncommitted_events.append(event)
    
    def _apply_event(self, event: DomainEvent):
        if isinstance(event, OrderPlacedEvent):
            self.status = OrderStatus.PLACED
        elif isinstance(event, ItemAddedEvent):
            self.items.append(event.item)
    
    def load_from_history(self, events: List[DomainEvent]):
        for event in events:
            self._apply_event(event)
```

### CQRS (Command Query Responsibility Segregation)

Separate read and write operations:

- **Commands**: Modify state (write)
- **Queries**: Retrieve data (read)

**Benefits:**
- Optimize read and write models separately
- Better performance for read-heavy systems
- Clear separation of concerns

```python
# Write Model
class CreateOrderCommand:
    def __init__(self, customer_id: str, items: List[dict]):
        self.customer_id = customer_id
        self.items = items

class OrderCommandHandler:
    def handle(self, command: CreateOrderCommand) -> str:
        order = OrderFactory.create_order(command.customer_id, command.items)
        self.order_repo.save(order)
        return order.order_id

# Read Model
class GetOrderQuery:
    def __init__(self, order_id: str):
        self.order_id = order_id

class OrderQueryHandler:
    def handle(self, query: GetOrderQuery) -> OrderDTO:
        # Query optimized read-only model
        order_data = self.order_read_model.find(query.order_id)
        return OrderDTO.from_data(order_data)
```

## Implementation Patterns

### Aggregate Design

**Guidelines:**
- Keep aggregates small
- Reference other aggregates by ID
- One aggregate per transaction
- Use domain events for cross-aggregate communication

### Entity Design

**Best Practices:**
- Use value objects for non-identity attributes
- Keep business logic in entities
- Use constructors to ensure valid state
- Never expose collections directly

### Repository Pattern

**Implementation Guidelines:**
- One repository per aggregate root
- Repository interface in domain layer
- Implementation in infrastructure layer
- Use domain language in repository methods

```python
# Domain Layer
class OrderRepository(ABC):
    @abstractmethod
    def save(self, order: Order): pass
    
    @abstractmethod
    def find_by_id(self, order_id: str) -> Order: pass
    
    @abstractmethod
    def find_pending_orders(self) -> List[Order]: pass

# Infrastructure Layer
class SqlOrderRepository(OrderRepository):
    def __init__(self, db_session):
        self.db_session = db_session
    
    def save(self, order: Order):
        db_order = self._to_db_model(order)
        self.db_session.merge(db_order)
        self.db_session.commit()
    
    def find_by_id(self, order_id: str) -> Order:
        db_order = self.db_session.query(DbOrder).filter_by(
            order_id=order_id
        ).first()
        return self._to_domain_model(db_order)
    
    def _to_domain_model(self, db_order) -> Order:
        # Convert database model to domain model
        pass
    
    def _to_db_model(self, order: Order):
        # Convert domain model to database model
        pass
```

### Domain-Driven Testing

**Testing Strategy:**

1. **Unit Tests**: Test domain logic in isolation
2. **Integration Tests**: Test aggregates with repositories
3. **Acceptance Tests**: Test complete scenarios

```python
import pytest

class TestOrder:
    def test_add_item_to_order(self):
        # Arrange
        order = Order("order-1", "customer-1")
        price = Money(100.00, "USD")
        
        # Act
        order.add_item("product-1", 2, price)
        
        # Assert
        assert len(order.items) == 1
        assert order.items[0].quantity == 2
    
    def test_cannot_add_item_with_negative_quantity(self):
        order = Order("order-1", "customer-1")
        price = Money(100.00, "USD")
        
        with pytest.raises(ValueError):
            order.add_item("product-1", -1, price)
    
    def test_cannot_place_empty_order(self):
        order = Order("order-1", "customer-1")
        
        with pytest.raises(ValueError):
            order.place_order()
```

## Common Pitfalls

### 1. Anemic Domain Model
**Problem**: Domain objects contain only data, business logic lives elsewhere.

**Solution**: Place business logic in domain objects, not in services.

```python
# Bad
class OrderService:
    def add_item_to_order(self, order: Order, product: Product, qty: int):
        if qty <= 0:
            raise ValueError("Invalid quantity")
        order.items.append(OrderItem(product, qty))

# Good
class Order:
    def add_item(self, product: Product, quantity: int):
        if quantity <= 0:
            raise ValueError("Invalid quantity")
        self.items.append(OrderItem(product, quantity))
```

### 2. Over-engineering
**Problem**: Creating overly complex domain models for simple problems.

**Solution**: Apply DDD principles proportionally to domain complexity. Simple domains don't need all DDD patterns.

### 3. Mixing Bounded Contexts
**Problem**: Allowing models from different contexts to bleed into each other.

**Solution**: Enforce context boundaries. Use separate packages/modules for each context.

### 4. Ignoring Domain Experts
**Problem**: Developers implementing their interpretation of the domain without domain expert input.

**Solution**: Maintain constant communication with domain experts. Evolve the ubiquitous language together.

### 5. Tightly Coupled Aggregates
**Problem**: Aggregates that reference each other directly, creating dependencies.

**Solution**: Reference aggregates by ID. Use domain events for communication.

## Best Practices

### 1. Start with the Domain
- Understand business rules before writing code
- Model the domain language explicitly
- Validate domain assumptions with experts

### 2. Maintain Ubiquitous Language
- Use domain terminology consistently everywhere
- Update documentation when language evolves
- Challenge vague or ambiguous terms
- Create glossaries for complex domains

### 3. Design for Change
- Keep aggregates focused and cohesive
- Use value objects for immutability
- Avoid tight coupling between bounded contexts
- Plan for subdomain evolution

### 4. Test Domain Logic
- Write tests for business rules
- Test at the domain level, not infrastructure
- Use specification-style tests
- Document expected behavior through tests

### 5. Refactor Ruthlessly
- Refactor domain models as understanding deepens
- Improve naming as the ubiquitous language evolves
- Simplify complex models
- Remove accidental complexity

### 6. Document Strategic Decisions
- Create context maps
- Document bounded context responsibilities
- Explain subdomain reasoning
- Maintain architecture decision records (ADRs)

### 7. Foster Domain-Driven Communication
- Regular design meetings with domain experts
- Use EventStorming for domain modeling
- Maintain design documentation
- Share learning across the team

## References

### Key Books
- **"Domain-Driven Design" by Eric Evans** - The foundational DDD book
- **"Implementing Domain-Driven Design" by Vaughn Vernon** - Practical DDD patterns
- **"Domain-Driven Design Distilled" by Vaughn Vernon** - Condensed DDD concepts

### Concepts
- Domain-Driven Design Official Website: https://domaindriven.org/
- Event Storming: https://eventstorming.com/
- CQRS Pattern: https://www.eventstore.com/blog/what-is-cqrs
- Context Mapping: Various bounded context patterns

### Related Patterns
- Clean Architecture
- Hexagonal Architecture (Ports & Adapters)
- Event Sourcing
- CQRS
- Vertical Slicing
- Microservices Architecture

---

## Conclusion

Domain Driven Design is not a silver bullet, but a powerful approach for building software that aligns with business reality. By placing the domain at the center of design and maintaining clear communication with domain experts, you create systems that are more maintainable, scalable, and ultimately more valuable to the business.

The key to successful DDD implementation is:
1. **Understanding** the domain deeply
2. **Collaborating** with domain experts
3. **Modeling** thoughtfully and refactoring continuously
4. **Keeping** the code aligned with business language
5. **Enforcing** bounded context boundaries

Start with the fundamentals, apply patterns pragmatically, and let your understanding of the domain guide your architecture.
