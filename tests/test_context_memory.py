#!/usr/bin/env python3
"""
Tests for the True Context Memory System.

These tests verify:
1. ContextBundle creation and serialization
2. Enhanced episodic memory operations
3. Vector index search
4. Session management and cross-session continuity
5. Context retrieval integration
"""

import sys
import time
import uuid
from pathlib import Path

# Add maven2_fix to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestContextBundle:
    """Tests for the ContextBundle data structure."""

    def test_create_empty_bundle(self):
        """Test creating an empty context bundle."""
        from brains.cognitive.memory_librarian.service.context_bundle import (
            ContextBundle,
            create_empty_bundle,
        )

        bundle = create_empty_bundle(
            user_id="test_user",
            session_id="test_session",
            source="tests",
        )

        assert bundle.user_id == "test_user"
        assert bundle.session_id == "test_session"
        assert bundle.metadata.source == "tests"
        assert not bundle.has_memories()
        assert not bundle.has_cache_hit()

    def test_bundle_serialization(self):
        """Test bundle serialization and deserialization."""
        from brains.cognitive.memory_librarian.service.context_bundle import (
            ContextBundle,
            RetrievedMemories,
        )

        bundle = ContextBundle(
            user_id="user123",
            session_id="sess456",
            current_episode_id="ep789",
        )

        # Add some memory hits
        bundle.retrieved_memories.stm_hits = [
            {"id": "1", "content": "test1"},
            {"id": "2", "content": "test2"},
        ]
        bundle.retrieved_memories.episodic_hits = [
            {"id": "3", "type": "episode", "summary": "test episode"},
        ]

        # Serialize
        data = bundle.to_dict()

        # Deserialize
        restored = ContextBundle.from_dict(data)

        assert restored.user_id == bundle.user_id
        assert restored.session_id == bundle.session_id
        assert restored.current_episode_id == bundle.current_episode_id
        assert len(restored.retrieved_memories.stm_hits) == 2
        assert len(restored.retrieved_memories.episodic_hits) == 1

    def test_bundle_memory_counts(self):
        """Test bundle memory counting."""
        from brains.cognitive.memory_librarian.service.context_bundle import (
            ContextBundle,
        )

        bundle = ContextBundle()

        assert bundle.retrieved_memories.count() == 0

        bundle.retrieved_memories.stm_hits = [{"id": "1"}, {"id": "2"}]
        bundle.retrieved_memories.vector_hits = [{"id": "3"}]

        assert bundle.retrieved_memories.count() == 3
        assert bundle.has_memories()

    def test_merge_bundles(self):
        """Test merging multiple bundles."""
        from brains.cognitive.memory_librarian.service.context_bundle import (
            ContextBundle,
            merge_bundles,
        )

        bundle1 = ContextBundle(user_id="user1")
        bundle1.retrieved_memories.stm_hits = [{"id": "1"}]

        bundle2 = ContextBundle(user_id="user1")
        bundle2.retrieved_memories.stm_hits = [{"id": "2"}]
        bundle2.retrieved_memories.episodic_hits = [{"id": "3"}]

        merged = merge_bundles([bundle1, bundle2])

        assert len(merged.retrieved_memories.stm_hits) == 2
        assert len(merged.retrieved_memories.episodic_hits) == 1


class TestEnhancedEpisodicMemory:
    """Tests for the enhanced episodic memory system."""

    def test_create_episode(self):
        """Test creating an episode."""
        from brains.memory.enhanced_episodic import (
            Episode,
            Turn,
        )

        episode = Episode(
            user_id="user123",
            session_id="sess456",
            topic="Testing episodic memory",
        )

        assert episode.user_id == "user123"
        assert episode.session_id == "sess456"
        assert episode.topic == "Testing episodic memory"
        assert not episode.is_closed
        assert episode.turn_count() == 0

    def test_add_turns_to_episode(self):
        """Test adding turns to an episode."""
        from brains.memory.enhanced_episodic import Episode

        episode = Episode()

        turn1 = episode.add_turn(
            question="What is Python?",
            answer="Python is a programming language.",
        )

        turn2 = episode.add_turn(
            question="What is it used for?",
            answer="It's used for many things.",
        )

        assert episode.turn_count() == 2
        assert turn1.question == "What is Python?"
        assert turn2.question == "What is it used for?"

    def test_close_episode(self):
        """Test closing an episode."""
        from brains.memory.enhanced_episodic import Episode

        episode = Episode()
        episode.add_turn("Q1", "A1")
        episode.add_turn("Q2", "A2")

        episode.close(summary="Test conversation")

        assert episode.is_closed
        assert episode.end_time is not None
        assert episode.summary == "Test conversation"

    def test_episode_serialization(self):
        """Test episode serialization."""
        from brains.memory.enhanced_episodic import Episode

        episode = Episode(
            user_id="user1",
            project_id="proj1",
            goal_ids=["goal1", "goal2"],
        )
        episode.add_turn("Q", "A")

        data = episode.to_dict()
        restored = Episode.from_dict(data)

        assert restored.user_id == episode.user_id
        assert restored.project_id == episode.project_id
        assert restored.goal_ids == episode.goal_ids
        assert restored.turn_count() == 1

    def test_episode_manager_lifecycle(self):
        """Test episode manager start/append/close."""
        from brains.memory.enhanced_episodic import (
            EpisodeManager,
        )

        manager = EpisodeManager()
        test_session = f"test_session_{uuid.uuid4()}"

        # Start episode
        episode = manager.start_episode(
            user_id="test_user",
            session_id=test_session,
            topic="Test topic",
        )

        assert episode is not None
        assert episode.topic == "Test topic"

        # Append turn
        turn = manager.append_turn(
            session_id=test_session,
            question="Test question",
            answer="Test answer",
        )

        assert turn is not None
        assert turn.question == "Test question"

        # Get current episode
        current = manager.get_current_episode(test_session)
        assert current is not None
        assert current.turn_count() == 1

        # Close episode
        closed = manager.close_episode(test_session, summary="Closed for test")
        assert closed is not None
        assert closed.is_closed
        assert closed.summary == "Closed for test"

    def test_query_episodes_by_time(self):
        """Test querying episodes by time range."""
        from brains.memory.enhanced_episodic import (
            EpisodeManager,
        )

        manager = EpisodeManager()

        # Create a test episode
        test_session = f"test_session_{uuid.uuid4()}"
        episode = manager.start_episode(
            user_id="test_user",
            session_id=test_session,
        )
        episode.add_turn("Q", "A")
        manager.close_episode(test_session)

        # Query for today
        results = manager.query_episodes(
            time_range="today",
            limit=10,
        )

        # Should find at least our test episode
        assert len(results) >= 1


class TestVectorIndex:
    """Tests for the vector index."""

    def test_create_vector_index(self):
        """Test creating a vector index."""
        from brains.memory.vector_index import VectorIndex

        index = VectorIndex(backend="memory")

        assert index is not None
        assert index.count() == 0

    def test_upsert_and_search(self):
        """Test inserting and searching documents."""
        from brains.memory.vector_index import VectorIndex

        index = VectorIndex(backend="memory")

        # Insert documents
        index.upsert("doc1", "Python is a programming language", {"type": "factual"})
        index.upsert("doc2", "Java is also a programming language", {"type": "factual"})
        index.upsert("doc3", "The weather is nice today", {"type": "casual"})

        assert index.count() == 3

        # Search for programming
        hits = index.search("Tell me about programming languages", top_k=2)

        assert len(hits) >= 1
        # The programming-related docs should score higher
        assert hits[0].id in ("doc1", "doc2")

    def test_search_with_filters(self):
        """Test search with metadata filters."""
        from brains.memory.vector_index import VectorIndex

        index = VectorIndex(backend="memory")

        index.upsert("doc1", "Python programming", {"type": "code"})
        index.upsert("doc2", "Python snake", {"type": "animal"})

        # Filter by type
        hits = index.search("Python", top_k=10, filters={"type": "code"})

        assert len(hits) >= 1
        assert all(h.metadata.get("type") == "code" for h in hits)

    def test_delete_document(self):
        """Test deleting a document."""
        from brains.memory.vector_index import VectorIndex

        index = VectorIndex(backend="memory")

        index.upsert("doc1", "Test document")
        assert index.count() == 1

        result = index.delete("doc1")
        assert result is True
        assert index.count() == 0

    def test_vector_hit_structure(self):
        """Test VectorHit data structure."""
        from brains.memory.vector_index import VectorHit

        hit = VectorHit(
            id="test_id",
            score=0.85,
            text="Test text",
            metadata={"key": "value"},
        )

        data = hit.to_dict()
        assert data["id"] == "test_id"
        assert data["score"] == 0.85
        assert data["text"] == "Test text"
        assert data["metadata"]["key"] == "value"


class TestSessionManager:
    """Tests for session management."""

    def test_create_session(self):
        """Test creating a session."""
        from brains.memory.session_manager import SessionManager

        manager = SessionManager()

        session = manager.start_session(source="tests")

        assert session is not None
        assert session.is_active
        assert session.source == "tests"

        # Clean up
        manager.end_session()

    def test_get_user_identity(self):
        """Test user identity persistence."""
        from brains.memory.session_manager import SessionManager

        manager = SessionManager()

        identity = manager.get_or_create_user_identity()

        assert identity is not None
        assert identity.user_id is not None

        # Should get same identity on second call
        identity2 = manager.get_or_create_user_identity()
        assert identity2.user_id == identity.user_id

    def test_record_turn(self):
        """Test recording a turn."""
        from brains.memory.session_manager import SessionManager

        manager = SessionManager()
        manager.start_session(source="tests")

        manager.record_turn("Test question", "Test answer")

        # Identity should have updated turn count
        assert manager._user_identity.total_turns >= 1

        manager.end_session()

    def test_session_lifecycle(self):
        """Test full session lifecycle."""
        from brains.memory.session_manager import (
            start_session,
            end_session,
            get_current_session,
            get_session_id,
        )

        # Start session
        session = start_session(source="tests")
        assert session is not None
        session_id = session.session_id

        # Get current session
        current = get_current_session()
        assert current is not None
        assert current.session_id == session_id

        # Get session ID
        retrieved_id = get_session_id()
        assert retrieved_id == session_id

        # End session
        ended = end_session()
        assert ended is not None
        assert not ended.is_active

        # Current session should be None
        assert get_current_session() is None

    def test_cross_session_context(self):
        """Test loading cross-session context."""
        from brains.memory.session_manager import SessionManager

        manager = SessionManager()

        # Create and end a session
        session1 = manager.start_session(source="tests")
        manager.record_turn("Q1", "A1")
        manager.end_session()

        # Create another session
        session2 = manager.start_session(source="tests")

        # Load cross-session context
        context = manager.load_cross_session_context()

        assert context is not None
        assert context["session_count"] >= 1

        manager.end_session()


class TestCapabilitySnapshot:
    """Tests for the capability snapshot system."""

    def test_self_scan(self):
        """Test running a self-scan."""
        from brains.system_capabilities import self_scan, CapabilitySnapshot

        snapshot = self_scan()

        assert snapshot is not None
        assert isinstance(snapshot, CapabilitySnapshot)
        assert snapshot.scan_time != ""

    def test_capability_snapshot_caching(self):
        """Test that capability snapshot is cached."""
        from brains.system_capabilities import (
            self_scan,
            get_capability_snapshot,
        )

        # Run initial scan
        snapshot1 = self_scan()

        # Get cached snapshot
        snapshot2 = get_capability_snapshot()

        # Should be the same object
        assert snapshot1.scan_time == snapshot2.scan_time

    def test_answer_memory_question(self):
        """Test answering memory-related questions."""
        from brains.system_capabilities import answer_memory_question

        # Test remembering question
        result = answer_memory_question("Can you remember what I said?")

        assert result is not None
        assert "answer" in result
        assert result["capability"] == "memory"

    def test_probe_vector_index(self):
        """Test vector index probe."""
        from brains.system_capabilities import probe_vector_index

        result = probe_vector_index()

        assert result is not None
        assert result.status is not None

    def test_probe_episodic_memory(self):
        """Test episodic memory probe."""
        from brains.system_capabilities import probe_episodic_memory

        result = probe_episodic_memory()

        assert result is not None
        assert result.status is not None


class TestContextRetrieval:
    """Tests for context retrieval integration."""

    def test_build_context_bundle(self):
        """Test building a context bundle from a query."""
        from brains.cognitive.memory_librarian.service.context_retrieval import (
            build_context_bundle,
        )

        bundle = build_context_bundle(
            query="What is Python?",
            user_id="test_user",
            session_id="test_session",
            source="tests",
        )

        assert bundle is not None
        assert bundle.user_id == "test_user"
        assert bundle.session_id == "test_session"
        assert bundle.metadata.query_used == "What is Python?"

    def test_retrieve_for_query(self):
        """Test quick retrieval for a query."""
        from brains.cognitive.memory_librarian.service.context_retrieval import (
            retrieve_for_query,
        )

        results = retrieve_for_query("test query", limit=5)

        assert isinstance(results, list)

    def test_answer_history_question(self):
        """Test answering history questions."""
        from brains.cognitive.memory_librarian.service.context_retrieval import (
            answer_history_question,
        )

        result = answer_history_question(
            query="What did we talk about yesterday?",
            user_id="test_user",
        )

        assert result is not None
        assert "answer" in result
        assert "found" in result
        assert "source" in result


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("RUNNING CONTEXT MEMORY TESTS")
    print("=" * 60)

    test_classes = [
        TestContextBundle,
        TestEnhancedEpisodicMemory,
        TestVectorIndex,
        TestSessionManager,
        TestCapabilitySnapshot,
        TestContextRetrieval,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  [PASS] {method_name}")
                    total_passed += 1
                except Exception as e:
                    print(f"  [FAIL] {method_name}: {str(e)[:60]}")
                    total_failed += 1
                    failures.append((f"{test_class.__name__}.{method_name}", str(e)))

    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    if failures:
        print("\nFailed tests:")
        for name, error in failures:
            print(f"  - {name}")
            print(f"    Error: {error[:100]}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
