import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter

class FinancialRAGSystem:
    def __init__(self, documents_df: pd.DataFrame):
        """
        Initialize RAG system with documents DataFrame
        
        Args:
            documents_df: DataFrame with columns ['doc_type', 'doc_info']
        """
        self.documents = documents_df.copy()
        
        # Clean and prepare document text
        self.documents['doc_info_clean'] = self.documents['doc_info'].fillna('').astype(str).apply(self._clean_text)
        
        # Initialize vectorizer with better parameters
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=10000,
            min_df=1,  # Reduced to capture more terms
            max_df=0.9,
            ngram_range=(1, 3),
            analyzer='word',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # Prepare documents for vectorization
        self.documents_text = self.documents['doc_info_clean'].tolist()
        
        # Create keyword index for semantic matching
        self._build_keyword_index()
        
        # Fit and transform documents
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents_text)
            self.feature_names = self.vectorizer.get_feature_names_out()
        except Exception as e:
            st.error(f"Vectorization error: {str(e)}")
            # Fallback to simple text matching
            self.tfidf_matrix = None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better matching"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\-\$\%\@\&\#\(\)\:]', ' ', text)
        
        # Expand common financial abbreviations
        financial_terms = {
            r'\bapr\b': 'annual percentage rate',
            r'\bapy\b': 'annual percentage yield',
            r'\broi\b': 'return on investment',
            r'\bfixed rate\b': 'fixed interest rate',
            r'\bvar rate\b': 'variable interest rate',
            r'\bmin pay\b': 'minimum payment',
            r'\blate fee\b': 'late payment fee',
            r'\bgrace period\b': 'grace period',
            r'\bdefault\b': 'loan default',
            r'\bmissed pay\b': 'missed payment',
            r'\bskip pay\b': 'skipped payment',
            r'\bdelinquent\b': 'delinquent payment',
            r'\boverdue\b': 'overdue payment',
            r'\bcredit score\b': 'credit score fico',
            r'\bfico\b': 'credit score',
        }
        
        for pattern, replacement in financial_terms.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _build_keyword_index(self):
        """Build a keyword index for semantic matching"""
        self.keyword_groups = {
            'missed_payment': [
                'miss', 'late', 'overdue', 'delinquent', 'default', 
                'skip', 'non-payment', 'failure to pay', 'past due',
                'unpaid', 'outstanding', 'arrears', 'penalty', 'fee'
            ],
            'payment_terms': [
                'payment', 'installment', 'due date', 'schedule',
                'monthly', 'biweekly', 'quarterly', 'automatic',
                'manual', 'wire', 'transfer', 'ach', 'direct debit'
            ],
            'fees': [
                'fee', 'charge', 'penalty', 'fine', 'cost',
                'late fee', 'overlimit', 'nsf', 'returned',
                'processing', 'service', 'administrative'
            ],
            'credit_impact': [
                'credit', 'score', 'report', 'bureau', 'fico',
                'transunion', 'equifax', 'experian', 'rating',
                'history', 'impact', 'affect', 'damage'
            ],
            'recovery': [
                'collection', 'recover', 'collect', 'agency',
                'legal', 'lawsuit', 'court', 'judgment',
                'garnishment', 'lien', 'repossess', 'foreclose'
            ]
        }
        
        # Create reverse mapping
        self.keyword_to_group = {}
        for group, keywords in self.keyword_groups.items():
            for keyword in keywords:
                self.keyword_to_group[keyword] = group
    
    def _semantic_expand_query(self, query: str) -> List[str]:
        """Expand query with semantically related terms"""
        query_clean = self._clean_text(query)
        query_words = query_clean.split()
        
        expanded_queries = [query_clean]
        
        # Check for keyword groups
        for word in query_words:
            if word in self.keyword_to_group:
                group = self.keyword_to_group[word]
                expanded_queries.append(' '.join(self.keyword_groups[group]))
        
        # Add specific expansions for common questions
        if any(term in query_clean for term in ['miss', 'late', 'overdue']):
            expanded_queries.extend([
                'late payment fee penalty charge',
                'missed payment grace period extension',
                'loan default consequences credit score impact',
                'what happens if you dont pay loan'
            ])
        
        if 'interest' in query_clean and 'rate' in query_clean:
            expanded_queries.extend([
                'annual percentage rate apr calculation',
                'fixed variable interest rate difference'
            ])
        
        return expanded_queries
    
    def search(self, query: str, k: int = 10, similarity_threshold: float = 0.01) -> List[Tuple[str, str, float]]:
        """Search for relevant documents using TF-IDF with semantic expansion"""
        if self.tfidf_matrix is None:
            # Fallback to keyword matching
            return self._keyword_search(query, k)
        
        # Expand query semantically
        expanded_queries = self._semantic_expand_query(query)
        
        all_results = []
        
        for expanded_query in expanded_queries:
            try:
                # Transform query
                query_vec = self.vectorizer.transform([expanded_query])
                
                # Calculate cosine similarities
                similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                
                # Get top indices
                valid_indices = np.where(similarities > similarity_threshold)[0]
                
                for idx in valid_indices:
                    doc_info = str(self.documents.iloc[idx]['doc_info'])
                    doc_type = str(self.documents.iloc[idx]['doc_type'])
                    similarity = float(similarities[idx])
                    
                    # Boost score for original query matches
                    if expanded_query == self._clean_text(query):
                        similarity *= 1.2  # Boost original query matches
                    
                    all_results.append((doc_type, doc_info, similarity, idx))
                    
            except Exception as e:
                continue
        
        # Remove duplicates by document index, keeping highest score
        unique_results = {}
        for doc_type, doc_info, similarity, idx in all_results:
            if idx not in unique_results or similarity > unique_results[idx][2]:
                unique_results[idx] = (doc_type, doc_info, similarity)
        
        # Convert to list and sort
        results = list(unique_results.values())
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:k]
    
    def _keyword_search(self, query: str, k: int = 10) -> List[Tuple[str, str, float]]:
        """Fallback keyword search when TF-IDF fails"""
        query_clean = self._clean_text(query)
        query_words = set(query_clean.split())
        
        scores = []
        
        for idx, row in self.documents.iterrows():
            doc_text = str(row['doc_info_clean'])
            doc_words = set(doc_text.split())
            
            # Calculate simple Jaccard similarity
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            
            if union > 0:
                similarity = intersection / union
                
                # Bonus for document type relevance
                doc_type_lower = str(row['doc_type']).lower()
                if any(term in doc_type_lower for term in ['loan', 'payment', 'policy', 'term']):
                    similarity *= 1.3
                
                if similarity > 0.01:
                    scores.append((idx, similarity))
        
        # Sort and get top k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, similarity in scores[:k]:
            doc_info = str(self.documents.iloc[idx]['doc_info'])
            doc_type = str(self.documents.iloc[idx]['doc_type'])
            results.append((doc_type, doc_info, similarity))
        
        return results
    
    def _find_related_info(self, query: str, max_chars: int = 300) -> str:
        """Find related information even without direct matches"""
        query_clean = self._clean_text(query)
        related_snippets = []
        
        # Look for financial consequences in all documents
        consequence_terms = ['fee', 'penalty', 'charge', 'interest', 'rate increase',
                           'credit report', 'collection', 'legal action', 'default']
        
        for idx, row in self.documents.iterrows():
            doc_text = str(row['doc_info_clean']).lower()
            doc_full = str(row['doc_info'])
            
            # Check for consequence terms
            if any(term in doc_text for term in consequence_terms):
                # Try to extract relevant sentences
                sentences = re.split(r'[.!?]+', doc_full)
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if (any(term in sentence_lower for term in consequence_terms) or
                        any(term in sentence_lower for term in ['if you', 'failure to', 'in the event'])):
                        
                        # Clean up the sentence
                        sentence = sentence.strip()
                        if len(sentence) > 20:  # Meaningful length
                            doc_type = str(row['doc_type'])
                            related_snippets.append(f"{doc_type}: {sentence}")
                            break
                
                if len(related_snippets) >= 3:  # Limit to 3 snippets
                    break
        
        if related_snippets:
            return " ".join(related_snippets)[:max_chars]
        
        return None
    
    def answer_question(self, query: str) -> str:
        """Generate answer based on retrieved documents"""
        # Search for relevant documents with lower threshold
        relevant_docs = self.search(query, k=8, similarity_threshold=0.001)
        
        # Extract meaningful information
        extracted_info = self._extract_financial_info(query, relevant_docs)
        
        if extracted_info:
            return self._generate_detailed_answer(query, extracted_info, relevant_docs)
        else:
            return self._generate_educated_response(query)
    
    def _extract_financial_info(self, query: str, relevant_docs: List[Tuple]) -> dict:
        """Extract structured financial information from documents"""
        query_clean = self._clean_text(query)
        info = {
            'fees': [],
            'timeframes': [],
            'consequences': [],
            'solutions': [],
            'contacts': [],
            'documents_found': []
        }
        
        for doc_type, doc_info, score in relevant_docs:
            if score > 0.05:  # Only use somewhat relevant documents
                info['documents_found'].append(f"{doc_type} (score: {score:.1%})")
                
                doc_text = str(doc_info).lower()
                
                # Extract fees
                fee_patterns = [
                    r'\$(\d+(?:\.\d{2})?)',  # $ amounts
                    r'(\d+(?:\.\d{2})?)%',   # percentages
                    r'fee of (\$?\d+(?:\.\d{2})?)',
                    r'charge.*?(\$?\d+(?:\.\d{2})?)',
                    r'penalty.*?(\$?\d+(?:\.\d{2})?)'
                ]
                
                for pattern in fee_patterns:
                    matches = re.findall(pattern, doc_text)
                    for match in matches:
                        if match not in info['fees']:
                            info['fees'].append(match)
                
                # Extract time-related info
                time_patterns = [
                    r'(\d+)\s*day', r'(\d+)\s*month', r'(\d+)\s*week',
                    r'within\s*(\d+)\s*day', r'after\s*(\d+)\s*day',
                    r'grace period.*?(\d+)', r'late after (\d+)'
                ]
                
                for pattern in time_patterns:
                    matches = re.findall(pattern, doc_text)
                    for match in matches:
                        if match not in info['timeframes']:
                            info['timeframes'].append(match)
                
                # Extract consequences
                if any(term in doc_text for term in ['credit report', 'credit score', 'fico']):
                    info['consequences'].append('Credit score impact')
                
                if any(term in doc_text for term in ['collection', 'agency', 'collect']):
                    info['consequences'].append('Sent to collections')
                
                if any(term in doc_text for term in ['legal', 'lawsuit', 'court']):
                    info['consequences'].append('Legal action')
                
                if any(term in doc_text for term in ['default', 'accelerate', 'due immediately']):
                    info['consequences'].append('Loan default')
                
                # Extract solutions
                if any(term in doc_text for term in ['contact', 'call', 'phone', 'email']):
                    # Extract contact info
                    contact_section = re.search(r'contact.*?(?:phone|call|email).*?(\d{3}[-.]?\d{3}[-.]?\d{4}|[\w\.-]+@[\w\.-]+)', 
                                               doc_text, re.IGNORECASE)
                    if contact_section:
                        info['contacts'].append(contact_section.group(1))
                
                if any(term in doc_text for term in ['payment plan', 'arrangement', 'extension']):
                    info['solutions'].append('Payment arrangements available')
        
        return info
    
    def _generate_detailed_answer(self, query: str, info: dict, relevant_docs: List[Tuple]) -> str:
        """Generate detailed answer from extracted information"""
        # Build answer sections
        sections = []
        
        # Introduction
        sections.append(f"**Your Question:** {query}")
        sections.append("")
        
        # Documents found
        if info['documents_found']:
            sections.append("**📄 Relevant Documents Found:**")
            for doc in info['documents_found'][:3]:  # Top 3
                sections.append(f"• {doc}")
            sections.append("")
        
        # Fees and charges
        if info['fees']:
            sections.append("**💰 Potential Fees/Charges:**")
            unique_fees = list(set(info['fees']))[:5]  # Limit to 5
            for fee in unique_fees:
                sections.append(f"• ${fee}" if fee.replace('.', '').isdigit() else f"• {fee}")
            sections.append("")
        
        # Timeframes
        if info['timeframes']:
            sections.append("**⏰ Important Timeframes:**")
            for timeframe in list(set(info['timeframes']))[:3]:
                sections.append(f"• {timeframe} days")
            sections.append("")
        
        # Consequences
        if info['consequences']:
            sections.append("**⚠️ Possible Consequences:**")
            for consequence in list(set(info['consequences'])):
                sections.append(f"• {consequence}")
            sections.append("")
        
        # Solutions
        if info['solutions']:
            sections.append("**✅ Available Solutions:**")
            for solution in list(set(info['solutions'])):
                sections.append(f"• {solution}")
            sections.append("")
        
        # Contacts (if any)
        if info['contacts']:
            sections.append("**📞 Contact Information:**")
            for contact in list(set(info['contacts']))[:2]:
                sections.append(f"• {contact}")
            sections.append("")
        
        # Summary based on query type
        query_lower = query.lower()
        summary = ""
        
        if any(term in query_lower for term in ['miss', 'late', 'overdue']):
            summary = """**Summary for Missed/Late Payments:**
1. **Immediate Action**: Contact your lender as soon as possible
2. **Fees**: Expect late payment fees (varies by lender)
3. **Grace Period**: Many loans have a 10-15 day grace period
4. **Credit Impact**: Payments >30 days late may affect credit score
5. **Options**: Payment plans or extensions may be available"""
        
        elif 'interest' in query_lower:
            summary = """**Summary on Interest Rates:**
Interest rates are determined by credit score, loan type, and market conditions. 
Fixed rates remain constant; variable rates may change."""
        
        else:
            summary = """**General Financial Guidance:**
Review your loan agreement for specific terms and conditions. 
Contact your financial institution for personalized information."""
        
        sections.append(summary)
        sections.append("")
        
        # Disclaimer
        sections.append("**📝 Important Note:**")
        sections.append("This information is based on document analysis. For accurate, personalized advice regarding your specific situation, contact your financial institution directly or consult with a qualified financial advisor.")
        
        return "\n".join(sections)
    
    def _generate_educated_response(self, query: str) -> str:
        """Generate an educated response when no documents match"""
        query_lower = query.lower()
        
        # Common financial scenarios with general advice
        scenarios = {
            'missed payment': """
**General Information About Missed Loan Payments:**

**Typical Consequences:**
1. **Late Fees**: Most lenders charge $25-$50 or 5% of payment
2. **Grace Period**: Usually 10-15 days before late fee applies
3. **Credit Reporting**: Payments 30+ days late reported to credit bureaus
4. **Interest**: Late payments may accrue additional interest
5. **Default Risk**: Multiple missed payments can lead to default

**Recommended Actions:**
1. **Contact Immediately**: Call your lender to explain situation
2. **Make Payment ASAP**: Minimize fees and credit impact
3. **Request Options**: Ask about payment plans or deferment
4. **Document Everything**: Keep records of all communications

**Document Review Needed:** 
Check your specific loan agreement for exact terms.""",
            
            'interest rate': """
**Understanding Interest Rates:**

**Common Rate Types:**
1. **Fixed Rates**: Remain constant throughout loan term
2. **Variable Rates**: Change based on market indexes
3. **Introductory Rates**: Lower initial rates that increase later

**Factors Affecting Rates:**
• Credit score and history
• Loan amount and term
• Collateral (secured vs unsecured)
• Current market conditions

**Check Your Documents For:**
• APR (Annual Percentage Rate)
• Rate adjustment terms
• Maximum rate caps (for variable rates)""",
            
            'loan approval': """
**Loan Approval Requirements:**

**Common Requirements:**
1. **Credit Score**: Typically 650+ for most loans
2. **Income Verification**: Pay stubs, tax returns, or bank statements
3. **Debt-to-Income Ratio**: Usually below 43%
4. **Collateral**: For secured loans
5. **Documentation**: ID, proof of address, employment verification

**Improving Approval Chances:**
• Improve credit score before applying
• Reduce existing debt
• Provide complete, accurate documentation
• Consider a co-signer if credit is limited"""
        }
        
        # Find matching scenario
        response = None
        for scenario, advice in scenarios.items():
            if scenario in query_lower:
                response = advice
                break
        
        if not response:
            # Generic financial advice
            response = f"""
**Financial Information for: "{query}"**

**Based on general financial knowledge:**

**Common Considerations:**
1. Review your specific loan or financial agreement documents
2. Contact your financial institution for exact terms
3. Check for grace periods, fees, and penalties
4. Understand impact on credit history and score

**Document Analysis Results:**
The system searched through available documents but did not find specific information matching your exact query.

**Suggested Next Steps:**
1. **Review Your Documents**: Check your loan agreement or policy documents
2. **Contact Support**: Reach out to your financial institution directly
3. **Try Different Terms**: Search for related terms like:
   • Fees and charges
   • Payment terms
   • Credit requirements
   • Policy details

**Available Document Types in System:** {', '.join(self.documents['doc_type'].unique()[:5])}"""
        
        return response


def financial_rag_system(cleaned_data):
    """
    Streamlit interface for Financial RAG Q&A System
    
    Args:
        cleaned_data: Dictionary containing cleaned dataframes including 'documents'
    """
    
    st.subheader("🤖 Financial Document Q&A Assistant")
    st.markdown("""
    Ask questions about financial policies, loan terms, regulations, or any 
    financial documentation in your system. The system will search through all
    available documents and provide relevant information.
    """)
    
    # Check if documents exist
    if 'documents' not in cleaned_data:
        st.error("❌ No documents data found. Please ensure documents are loaded and cleaned in previous steps.")
        st.info("Make sure your documents CSV file has columns: 'doc_type' and 'doc_info'")
        
        # Create sample data for testing
        st.markdown("### For testing purposes, you can use sample data:")
        if st.button("Load Sample Documents", key="load_sample"):
            sample_data = {
                'doc_type': [
                    'Loan Policy Document',
                    'Credit Requirements Guide', 
                    'Interest Rate Schedule',
                    'Payment Terms Agreement',
                    'Late Payment Policy',
                    'Default and Collections',
                    'Fee Schedule Document',
                    'Credit Impact Guidelines'
                ],
                'doc_info': [
                    'Loan applications require minimum credit score 650. Late payments incur $35 fee after 15-day grace period. Multiple late payments may result in increased interest rates or default proceedings. Contact loan servicing at 1-800-LOAN-NOW for payment arrangements.',
                    'Personal loans require proof of income, identification verification, and credit check. Minimum debt-to-income ratio 40%. Late payments reported to credit bureaus after 30 days delinquency.',
                    'Current interest rates: Prime + 3.99% to 15.99% based on creditworthiness. Rates may increase for accounts with late payment history. Variable rates adjust quarterly.',
                    'Monthly payments due on 1st of each month. Late fee $35 applied after 15-day grace period. Payments 60+ days overdue may be sent to collections. Payment extensions available with 48-hour notice.',
                    'Late payment policy: $35 fee for payments 1-15 days late, $50 fee for payments 16-30 days late. Payments over 30 days late reported to credit bureaus. Accounts 90+ days overdue considered in default.',
                    'Default procedures begin after 90 days delinquency. Account may be sent to collections agency. Legal action may be pursued for balances over $10,000. Repossession possible for secured loans.',
                    'Fee schedule: Late payment $35, Returned payment $25, Payment extension $15, Document copy $10 per page. All fees subject to change with 30-day notice.',
                    'Credit reporting: Late payments reported to Equifax, Experian, TransUnion after 30 days. Credit score may decrease 40-100 points for serious delinquency. Good payment history for 24 months can help rebuild score.'
                ]
            }
            cleaned_data['documents'] = pd.DataFrame(sample_data)
            st.success("✅ Sample documents loaded! Contains specific information about late payments, fees, and consequences.")
            st.rerun()
        return
    
    documents_df = cleaned_data['documents']
    
    # Validate and clean data
    if 'doc_type' not in documents_df.columns or 'doc_info' not in documents_df.columns:
        st.error("❌ Required columns 'doc_type' and 'doc_info' not found in documents.")
        st.info(f"Available columns: {list(documents_df.columns)}")
        return
    
    # Clean data
    documents_df = documents_df.dropna(subset=['doc_type', 'doc_info'])
    documents_df['doc_type'] = documents_df['doc_type'].fillna('Unknown').astype(str)
    documents_df['doc_info'] = documents_df['doc_info'].fillna('').astype(str)
    
    if len(documents_df) == 0:
        st.warning("⚠️ No documents available after cleaning.")
        return
    
    # Display document stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Total Documents", len(documents_df))
    with col2:
        unique_types = documents_df['doc_type'].nunique()
        st.metric("📊 Document Types", unique_types)
    with col3:
        avg_length = documents_df['doc_info'].str.len().mean()
        st.metric("📏 Avg. Length", f"{avg_length:.0f} chars")
    
    # Display top document types
    st.markdown("**Top Document Types:**")
    doc_type_counts = documents_df['doc_type'].value_counts().head(5)
    for doc_type, count in doc_type_counts.items():
        st.markdown(f"- **{doc_type}**: {count} documents")
    
    # Initialize RAG system
    try:
        with st.spinner("🔄 Initializing Financial Q&A System..."):
            rag = FinancialRAGSystem(documents_df)
        st.success("✅ Financial Q&A system ready!")
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        return
    
    st.markdown("---")
    
    # Initialize session state
    if 'rag_query' not in st.session_state:
        st.session_state.rag_query = ""
    
    # Example questions with better organization
    st.markdown("### 💡 Common Financial Questions:")
    
    # Categorize examples
    example_categories = {
        "Payment Issues": [
            "What happens if I miss a loan payment?",
            "How much is the late payment fee?",
            "What is the grace period for payments?",
            "Can I get a payment extension?"
        ],
        "Loan Information": [
            "What are the requirements for loan approval?",
            "How is interest rate calculated?",
            "What is the minimum credit score required?",
            "What documents are needed for application?"
        ],
        "Credit & Consequences": [
            "How do late payments affect my credit score?",
            "When are late payments reported to credit bureaus?",
            "What happens if I default on a loan?",
            "How can I rebuild my credit after late payments?"
        ]
    }
    
    # Display examples in expandable sections
    for category, examples in example_categories.items():
        with st.expander(f"📋 {category} Questions", expanded=False):
            cols = st.columns(2)
            for i, example in enumerate(examples):
                col_idx = i % 2
                if cols[col_idx].button(
                    example[:45] + "..." if len(example) > 45 else example,
                    key=f"ex_{category}_{i}",
                    use_container_width=True
                ):
                    st.session_state.rag_query = example
    
    st.markdown("---")
    
    # Main query input
    st.markdown("### 🔍 Ask Your Financial Question:")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Type your question here:",
            value=st.session_state.rag_query,
            placeholder="e.g., What are the penalties for late loan payments?",
            key="rag_input",
            label_visibility="collapsed"
        )
    
    with col2:
        search_clicked = st.button("🔍 Search Documents", type="primary", use_container_width=True)
    
    # Process query
    if search_clicked and query and query.strip():
        st.session_state.rag_query = query
        
        with st.spinner("🔍 Analyzing financial documents..."):
            # Get answer
            answer = rag.answer_question(query)
            
            # Display answer
            st.markdown("### 📊 Analysis Results")
            st.markdown("---")
            
            # Split answer into sections for better display
            answer_lines = answer.split('\n')
            
            for line in answer_lines:
                if line.startswith('**') and line.endswith('**'):
                    st.markdown(f"#### {line}")
                elif line.strip():
                    st.markdown(line)
                else:
                    st.markdown("")  # Empty line
            
            st.markdown("---")
            
            # Show matching documents with scores
            st.markdown("### 📑 Document Matches Found")
            
            relevant_docs = rag.search(query, k=6, similarity_threshold=0.001)
            
            if relevant_docs:
                # Create tabs for different views
                tab1, tab2 = st.tabs(["📊 By Relevance Score", "📋 Document List"])
                
                with tab1:
                    # Display as bars
                    for i, (doc_type, doc_info, score) in enumerate(relevant_docs, 1):
                        if score > 0.01:  # Only show meaningful matches
                            # Progress bar visualization
                            st.markdown(f"**{doc_type}**")
                            st.progress(
                                min(score * 2, 1.0),  # Scale for visibility
                                text=f"Relevance: {score:.1%}"
                            )
                            
                            # Brief preview
                            preview = doc_info[:100] + "..." if len(doc_info) > 100 else doc_info
                            st.caption(f"*{preview}*")
                            st.markdown("---")
                
                with tab2:
                    # Detailed list view
                    for i, (doc_type, doc_info, score) in enumerate(relevant_docs, 1):
                        with st.expander(f"Document {i}: {doc_type} (Score: {score:.1%})", 
                                       expanded=(i == 1 and score > 0.1)):
                            st.markdown(f"**Type:** {doc_type}")
                            st.markdown(f"**Relevance:** {score:.1%}")
                            st.markdown("**Content:**")
                            st.write(doc_info)
            else:
                st.info("""
                **Document Search Results:**
                No documents with high similarity found. 
                
                **Why this might happen:**
                1. Your question uses different terminology than the documents
                2. The specific information isn't in the available documents
                3. Try rephrasing or using more general terms
                
                **The answer above is based on:**
                • General financial knowledge patterns
                • Related information found in documents
                • Common financial practices
                """)
    
    elif query and query.strip():
        st.info("👆 Click 'Search Documents' to analyze your question")
    
    else:
        st.info("💡 Enter a question or click an example above to begin")
    
    # Quick tips
    with st.expander("💡 Tips for Better Financial Questions", expanded=False):
        st.markdown("""
        **For accurate information about:**
        
        **Late/Missed Payments:**
        • "late payment fees"
        • "grace period for payments"
        • "credit score impact of late payments"
        • "payment extension options"
        
        **Loan Terms:**
        • "interest rate calculation"
        • "loan approval requirements"
        • "minimum credit score"
        • "document requirements"
        
        **General Financial:**
        • "fee schedule"
        • "payment methods"
        • "contact information"
        • "policy documents"
        
        **Example effective questions:**
        • "What are the fees for late loan payments?"
        • "How long is the grace period before late fees?"
        • "When are late payments reported to credit bureaus?"
        • "What options exist for missed payments?"
        """)
    
    # Sample document preview
    with st.expander("📂 Preview Sample Documents", expanded=False):
        st.markdown("**First 3 documents in system:**")
        
        for idx, row in documents_df.head(3).iterrows():
            st.markdown(f"**{row['doc_type']}**")
            
            # Clean up display
            content = str(row['doc_info'])
            if len(content) > 200:
                content = content[:200] + "..."
            
            st.markdown(f"> *{content}*")
            st.markdown("---")

# Test function (can be removed in production)
def test_rag_system():
    """Test the RAG system with sample data"""
    test_data = {
        'doc_type': [
            'Late Payment Policy',
            'Loan Agreement Terms',
            'Fee Schedule',
            'Credit Reporting Policy',
            'Collections Procedure'
        ],
        'doc_info': [
            'Late payments incur a $35 fee if not received within 15 days of due date. Payments over 30 days late will be reported to credit bureaus. Multiple late payments may result in increased interest rates or loan default.',
            'All loans subject to credit approval. Minimum credit score 650 required. Interest rates vary from 5.99% to 19.99% APR based on creditworthiness.',
            'Fee schedule: Late payment $35, Returned check $25, Payment extension $15, Document copy $10 per page.',
            'Credit reporting: Late payments reported after 30 days delinquency to Equifax, Experian, and TransUnion. Serious delinquency may reduce credit score by 40-100 points.',
            'Accounts 90+ days overdue sent to collections. Legal action may be pursued for balances exceeding $10,000. Contact collections department at 1-800-COLLECT for arrangements.'
        ]
    }
    
    df = pd.DataFrame(test_data)
    rag = FinancialRAGSystem(df)
    
    # Test queries
    test_queries = [
        "What happens if I miss a loan payment?",
        "How much is the late fee?",
        "When are late payments reported to credit bureaus?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        answer = rag.answer_question(query)
        print(answer)

if __name__ == "__main__":
    # Run test if needed
    test_rag_system()