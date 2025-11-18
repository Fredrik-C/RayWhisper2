"""RAG (Retrieval-Augmented Generation) service."""

import re

import tiktoken
from loguru import logger

from ...domain.interfaces.vector_store import IVectorStore
from ...infrastructure.embeddings.reranker import CrossEncoderReranker


class RAGService:
    """Service for retrieving and reranking context for RAG."""

    # Common English words to exclude from technical term extraction
    _COMMON_WORDS = {
        # Articles, pronouns, conjunctions
        "The", "This", "That", "These", "Those", "A", "An", "And", "Or", "But",
        "For", "With", "From", "To", "In", "On", "At", "By", "As", "Of",
        "About", "After", "Before", "During", "Through", "Over", "Under",
        "Between", "Among", "Into", "Onto", "Upon", "Within", "Without",
        "Example", "Examples", "Usage", "Note", "Notes", "See", "Also",
        "More", "Less", "Most", "Least", "Some", "All", "Any", "Each",
        "Every", "Both", "Either", "Neither", "Other", "Another", "Such",
        # Generic technical words that add noise
        "Cloud", "Open", "Source", "Database", "Store", "Technologies",
        "Vector", "Search", "System", "Library", "Framework", "Platform",
        "Service", "Application", "Data", "Information", "Content",
        "Technology", "Software", "Code", "File", "Files", "Folder",
        "Directory", "Path", "Name", "Type", "Value", "Object", "Class",
        "Method", "Function", "Property", "Field", "Parameter", "Return",
        "String", "Number", "Boolean", "Array", "List", "Dictionary",
        "Collection", "Set", "Map", "Queue", "Stack", "Tree", "Graph",
        "Node", "Edge", "Link", "Reference", "Pointer", "Address",
        "Memory", "Cache", "Buffer", "Stream", "Reader", "Writer",
        "Input", "Output", "Request", "Response", "Query", "Result",
        "Error", "Exception", "Warning", "Message", "Log", "Debug",
        "Test", "Testing", "Unit", "Integration", "End", "Start",
        "Create", "Read", "Update", "Delete", "Get", "Set", "Add",
        "Remove", "Insert", "Select", "Where", "Order", "Group",
        "Join", "Union", "Intersect", "Except", "Distinct", "Count",
        "Sum", "Average", "Min", "Max", "First", "Last", "Single",
        "Default", "Empty", "Null", "True", "False", "Yes", "No",
    }

    def __init__(
        self,
        vector_store: IVectorStore,
        reranker: CrossEncoderReranker,
        top_k: int = 5,
        top_n: int = 3,
    ) -> None:
        """Initialize the RAG service.

        Args:
            vector_store: The vector store to search.
            reranker: The reranker to use for improving results.
            top_k: Number of results to retrieve from vector store.
            top_n: Number of results to keep after reranking.
        """
        self._vector_store = vector_store
        self._reranker = reranker
        self._top_k = top_k
        self._top_n = top_n
        
        # Initialize tokenizer for context length management
        # Whisper uses GPT-2 tokenizer
        self._tokenizer = tiktoken.get_encoding("gpt2")

    # Phonetic substitution rules for common speech-to-text errors
    _PHONETIC_SUBSTITUTIONS = {
        # Numbers to words
        '0': ['zero', 'oh'],
        '1': ['one', 'won'],
        '2': ['to', 'too', 'two'],
        '3': ['three', 'tree'],
        '4': ['for', 'four', 'fore'],
        '5': ['five'],
        '6': ['six'],
        '7': ['seven'],
        '8': ['ate', 'eight', 'iate'],
        '9': ['nine'],
        # Letter combinations
        'vv': ['w', 'double-v', 'doublev', 'weav'],
        'w': ['vv', 'double-v', 'doublev'],
        'uu': ['w', 'double-u', 'doubleu'],
        # Common phonetic confusions
        'c': ['k', 's'],
        'k': ['c'],
        's': ['c', 'z'],
        'z': ['s'],
        'f': ['ph'],
        'ph': ['f'],
    }

    @staticmethod
    def _apply_phonetic_substitutions(word: str) -> list[str]:
        """Apply phonetic substitution rules to generate variations of a word.

        Args:
            word: The word to generate variations for.

        Returns:
            List of phonetic variations including the original word.
        """
        variations = {word}  # Start with original
        word_lower = word.lower()

        # Apply substitutions for each pattern found in the word
        for pattern, replacements in RAGService._PHONETIC_SUBSTITUTIONS.items():
            if pattern in word_lower:
                for replacement in replacements:
                    # Replace the pattern with each possible replacement
                    variation = word_lower.replace(pattern, replacement)
                    variations.add(variation)

        return list(variations)

    @staticmethod
    def _get_adaptive_threshold(word_length: int) -> float:
        """Get adaptive similarity threshold based on word length.

        Shorter words need lower thresholds because small differences
        have larger impact on similarity scores.

        Args:
            word_length: Length of the word.

        Returns:
            Similarity threshold (0.0 to 1.0).
        """
        if word_length <= 3:
            return 0.35  # Very short words (e.g., "VV8")
        elif word_length <= 5:
            return 0.45  # Short words
        elif word_length <= 7:
            return 0.55  # Medium words
        else:
            return 0.65  # Long words

    @staticmethod
    def _calculate_phonetic_similarity(word1: str, word2: str) -> float:
        """Calculate phonetic similarity between two words using multiple strategies.

        Args:
            word1: First word.
            word2: Second word.

        Returns:
            Similarity score (0.0 to 1.0).
        """
        from difflib import SequenceMatcher

        word1_lower = word1.lower()
        word2_lower = word2.lower()

        # Calculate length ratio to penalize very different lengths
        len1, len2 = len(word1_lower), len(word2_lower)
        length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0

        # Strategy 1: Direct string similarity
        direct_similarity = SequenceMatcher(None, word1_lower, word2_lower).ratio()

        # Strategy 2: Apply phonetic substitutions and check similarity
        word1_variations = RAGService._apply_phonetic_substitutions(word1_lower)
        word2_variations = RAGService._apply_phonetic_substitutions(word2_lower)

        max_substitution_similarity = 0.0
        for var1 in word1_variations:
            for var2 in word2_variations:
                sim = SequenceMatcher(None, var1, var2).ratio()
                max_substitution_similarity = max(max_substitution_similarity, sim)

        # Strategy 3: Substring matching with stricter rules
        # Only apply bonus if:
        # 1. The substring is substantial (at least 40% of the shorter word)
        # 2. The length ratio is reasonable (at least 0.4)
        substring_bonus = 0.0
        if length_ratio >= 0.4:  # Words must be somewhat similar in length
            if word1_lower in word2_lower or word2_lower in word1_lower:
                # Calculate what percentage of the shorter word is the substring
                shorter_len = min(len1, len2)
                if shorter_len >= 4:  # Only for words of reasonable length
                    substring_bonus = 0.2  # Reduced from 0.3

        # Combine strategies with length ratio penalty
        base_score = max(
            direct_similarity,
            max_substitution_similarity * 0.95,  # Slight penalty for substitutions
        )

        # Apply length ratio as a multiplier (penalize very different lengths)
        # For similar lengths (ratio close to 1.0), this has minimal effect
        # For very different lengths (ratio close to 0), this significantly reduces the score
        length_penalty = 0.5 + (length_ratio * 0.5)  # Range: 0.5 to 1.0

        combined_score = (base_score * length_penalty) + substring_bonus

        return min(1.0, combined_score)  # Cap at 1.0

    @staticmethod
    def _find_phonetic_matches(query: str, technical_terms: set[str]) -> list[str]:
        """Find technical terms that are phonetically similar to words in the query.

        Uses multi-strategy phonetic matching including:
        - Phonetic substitution rules (e.g., "8" -> "ate", "VV" -> "W")
        - Adaptive similarity thresholds based on word length
        - Multiple similarity calculation strategies
        - Minimum word length filtering to avoid false positives

        Args:
            query: The transcribed query text.
            technical_terms: Set of technical terms to match against.

        Returns:
            List of terms that are phonetically similar to query words.
        """
        # Minimum word length to consider for phonetic matching
        # This prevents common short words like "we", "it", "re" from matching
        MIN_WORD_LENGTH = 3

        query_words = query.lower().split()
        similar_terms = []

        for term in technical_terms:
            term_lower = term.lower()
            max_similarity = 0.0
            best_word = ""

            for word in query_words:
                # Skip very short words to avoid false positives
                # Exception: if the word contains numbers, allow it (e.g., "vv8", "8")
                has_digit = any(c.isdigit() for c in word)
                if len(word) < MIN_WORD_LENGTH and not has_digit:
                    continue

                # Calculate phonetic similarity using enhanced algorithm
                similarity = RAGService._calculate_phonetic_similarity(word, term_lower)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_word = word

            # Use adaptive threshold based on the length of the query word
            if best_word:
                threshold = RAGService._get_adaptive_threshold(len(best_word))

                if max_similarity >= threshold:
                    similar_terms.append((term, max_similarity, best_word))
                    logger.debug(
                        f"Phonetic match: '{best_word}' -> '{term}' "
                        f"(similarity={max_similarity:.3f}, threshold={threshold:.3f})"
                    )

        # Calculate ranking scores with priority for misheard words
        ranked_terms = []
        for term, similarity, query_word in similar_terms:
            term_length = len(term)
            query_has_digit = any(c.isdigit() for c in query_word)
            query_has_unusual_pattern = query_has_digit or len(query_word) <= 3

            # Priority 1: Matches from likely misheard words (with digits/short unusual words)
            # These are the words we most want to correct
            priority = 0
            if query_has_digit and term_length >= 6:
                priority = 3  # Highest priority for digit queries matching long terms (e.g., "vv8" -> "Weaviate")
            elif query_has_digit:
                priority = 2  # Medium-high priority for digit queries matching short terms
            elif query_has_unusual_pattern and term_length >= 6:
                priority = 2  # High priority for short queries matching long terms
            elif term_length >= 8:
                priority = 1  # Medium priority for long technical terms

            # Similarity score (0-1 range)
            score = similarity

            # Length bonus for technical terms
            length_bonus = 0
            if term_length >= 8:
                length_bonus = 0.15
            elif term_length >= 6:
                length_bonus = 0.10
            elif term_length >= 4:
                length_bonus = 0.05

            # Combine: priority (0-3) * 10 + score (0-1) + length_bonus (0-0.15)
            # This ensures priority dominates, then similarity, then length
            final_score = (priority * 10) + score + length_bonus

            ranked_terms.append((term, final_score, similarity, query_word, priority))

        # Sort by final score (highest first)
        ranked_terms.sort(key=lambda x: x[1], reverse=True)

        # Return top 5 high-quality matches
        return [term for term, _, _, _, _ in ranked_terms[:5]]

    @staticmethod
    def _extract_technical_terms(text: str) -> set[str]:
        """Extract likely technical terms from text using heuristics.

        Looks for:
        - Capitalized words (proper nouns, product names)
        - CamelCase identifiers
        - Dot-separated identifiers (e.g., "RichardSzalay.MockHttp")
        - All-caps acronyms
        - Mixed-case technical names (e.g., "NSubstitute", "AutoMapper")

        Args:
            text: Text to extract terms from.

        Returns:
            Set of extracted technical terms.
        """
        terms = set()

        # Pattern 1: Dot-separated identifiers (e.g., "RichardSzalay.MockHttp", "System.IO")
        dot_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:\.[A-Z][a-zA-Z0-9]*)+\b'
        terms.update(re.findall(dot_pattern, text))

        # Pattern 2: Words with multiple capitals (CamelCase, PascalCase, or mixed like "NSubstitute")
        # Matches words that start with capital and have at least one more capital letter
        multi_cap_pattern = r'\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b'
        terms.update(re.findall(multi_cap_pattern, text))

        # Pattern 3: All-caps acronyms (2+ letters, e.g., "FAISS", "API", "HTTP")
        acronym_pattern = r'\b[A-Z]{2,}\b'
        terms.update(re.findall(acronym_pattern, text))

        # Pattern 4: Capitalized words (but filter out common words)
        capitalized_pattern = r'\b[A-Z][a-z]+\b'
        capitalized = re.findall(capitalized_pattern, text)
        terms.update(word for word in capitalized if word not in RAGService._COMMON_WORDS)

        return terms

    def retrieve_context(self, query: str, max_tokens: int = 100, condensed: bool = False, return_hotwords: bool = False):
        """Retrieve and rerank context for a query.

        Args:
            query: The search query.
            max_tokens: Maximum tokens for the returned context (defaults to 100).
            condensed: If True, return a condensed context (fewer terms/snippets).

        Returns:
            str: Formatted context string combining top reranked results.
        """
        logger.debug(f"Retrieving context for query: '{query[:50]}...'")

        # Initial retrieval from vector store
        results = self._vector_store.search(query, top_k=self._top_k)

        if not results:
            logger.debug("No results found in vector store")
            return ""

        logger.debug(f"Retrieved {len(results)} initial results")

        # Rerank results
        reranked = self._reranker.rerank(query, results, top_n=self._top_n)

        if not reranked:
            logger.debug("No results after reranking")
            return ""

        logger.debug(f"Reranked to {len(reranked)} results")

        # Log retrieved content for debugging
        for i, r in enumerate(reranked[:3]):  # Log top 3
            logger.info(
                f"Retrieved chunk {i+1}: score={r.score:.3f}, "
                f"content='{r.content[:100]}...', "
                f"header='{r.metadata.get('header_raw', 'N/A')}'"
            )

        # Format context optimized for Whisper's initial_prompt
        # Extract technical terms with relevance scoring
        term_scores: dict[str, float] = {}

        for i, r in enumerate(reranked):
            # Extract technical terms from the content using heuristics
            terms = self._extract_technical_terms(r.content)

            # Also extract from header if available
            header_raw = r.metadata.get('header_raw', '')
            if header_raw:
                header_terms = self._extract_technical_terms(header_raw)
                terms.update(header_terms)

            # Score terms based on chunk relevance and position
            # Higher score for terms in more relevant chunks (higher rerank score)
            # Higher score for terms in earlier positions
            position_weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, 0.25, 0.2
            chunk_weight = r.score if r.score > 0 else 0.5
            weight = position_weight * chunk_weight

            for term in terms:
                term_scores[term] = term_scores.get(term, 0) + weight

        # Find phonetically similar terms to the query
        phonetic_matches = self._find_phonetic_matches(query, set(term_scores.keys()))

        if phonetic_matches:
            logger.info(f"Phonetic matches for '{query}': {phonetic_matches}")

        # Boost scores for phonetically similar terms moderately
        # This helps them surface but avoids overpowering the reranker
        for term in phonetic_matches:
            if term in term_scores:
                term_scores[term] *= 8.0
            else:
                # If term wasn't in the extracted terms, add it with a reasonable base score
                term_scores[term] = 16.0

        # Build final context: technical terms for exact spelling guidance
        # For condensed mode (long audio) produce a compact, natural-language
        # instruction that emphasizes exact spellings, casing and punctuation.
        context_parts = []

        # Always include ALL phonetic matches first with directive language
        if phonetic_matches:
            # Keep phonetic matches for hotwords and optional non-condensed contexts
            top_match = phonetic_matches[0]
            context_parts.append(f"{top_match}. {top_match}. {top_match}")
            spell_directive = ", ".join(phonetic_matches)
            context_parts.append(f"Spell: {spell_directive}")

        if term_scores:
            term_limit = 3 if condensed else 8
            ranked_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            top_terms = [term for term, score in ranked_terms[:term_limit]]
            other_terms = [term for term in top_terms if term not in phonetic_matches]
            if other_terms and not condensed:
                terms_list = ", ".join(other_terms)
                context_parts.append(f"Technical terms: {terms_list}")

        # For condensed mode, synthesize a short instruction rather than a terse token list
        if condensed:
            important = []
            if phonetic_matches:
                important.extend(phonetic_matches)
            if term_scores:
                # Use the previously computed top_terms as additional important terms
                ranked_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
                top_terms_for_condensed = [term for term, _ in ranked_terms[:3]]
                for t in top_terms_for_condensed:
                    if t not in important:
                        important.append(t)

            # Deduplicate while preserving order
            seen = set()
            important_ordered = []
            for t in important:
                if t not in seen:
                    seen.add(t)
                    important_ordered.append(t)

            if important_ordered:
                proper_list = ", ".join(important_ordered)
                # Natural-language instruction to preserve punctuation/casing and prefer these spellings
                # Also provide short quoted examples and explicit punctuation guidance
                examples = ", ".join([f"'{t}'" for t in important_ordered[:10]])
                context = (
                    f"Use these exact spellings and capitalizations when transcribing: {proper_list}. "
                    f"Examples: {examples}. "
                    "Prioritize correct punctuation and capitalization. "
                    "Use commas to separate list items and end sentences with periods. "
                    "Preserve names and product spellings exactly as provided."
                )
            else:
                context = (
                    "Prioritize correct punctuation and capitalization in the transcription. "
                    "Preserve names and technical terms where possible."
                )
        else:
            # Non-condensed: join collected context parts
            context = ". ".join(context_parts)

        # CRITICAL: Whisper's initial_prompt has a hard limit of ~224 tokens
        # (n_text_ctx // 2 where n_text_ctx = 448)
        # We must truncate the context to avoid consuming tokens needed for
        # the actual transcription output, which would cause truncation
        
        tokens = self._tokenizer.encode(context)
        if len(tokens) > max_tokens:
            # Truncate to fit within token limit
            truncated_tokens = tokens[:max_tokens]
            context = self._tokenizer.decode(truncated_tokens)
            logger.warning(
                f"RAG context truncated from {len(tokens)} to {max_tokens} tokens "
                f"to stay within Whisper's initial_prompt limit"
            )

        mode = "condensed" if condensed else "full"
        logger.info(
            f"Generated RAG context [{mode}] ({len(tokens)} tokens, {len(context)} chars): '{context[:100]}...' "
            f"({len(term_scores)} terms extracted, {len(phonetic_matches)} phonetic matches)"
        )

        # Build a hotwords string (space-separated) from phonetic matches + top terms
        hotwords_list = []
        if phonetic_matches:
            hotwords_list.extend(phonetic_matches)

        if term_scores:
            ranked_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            top_terms_for_hotwords = [term for term, _ in ranked_terms[:3]]
            for t in top_terms_for_hotwords:
                if t not in hotwords_list:
                    hotwords_list.append(t)

        # Expand hotwords conservatively: include original and lowercase only.
        # Avoid adding title-cased variants which can force unnatural capitalization.
        expanded = []
        for w in hotwords_list:
            if w not in expanded:
                expanded.append(w)
            lw = w.lower()
            if lw not in expanded:
                expanded.append(lw)

        hotwords = " ".join(expanded) if expanded else None

        if return_hotwords:
            return context, hotwords

        return context

