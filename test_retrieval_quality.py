#!/usr/bin/env python3
"""
Retrieval Quality Testing Script

This script tests the retrieval system with a set of predefined questions
and evaluates the quality of responses.
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any
import requests

# Test questions with difficulty levels
questions = [
    (1, "易", "113 年度的總經理是誰？"),
    (2, "易", "113 年度總經理 A+B+C+D 四項酬金總額為多少？佔稅後純益的比例為多少？"),
    (3, "易", "公司 113 年度個體稅後純益（仟元）為多少？"),
    (4, "易", "副總經理職務在 113 年度是否有人任職（是否空缺）？"),
    (5, "易", "113 年度董事會共召開幾次會議？"),
    (6, "易", "哪位獨立董事的出席率為 80%？其實際出席與委託出席次數分別為多少？"),
    (7, "易", "請列出董事會出席情形表中的所有獨立董事姓名。"),
    (8, "中", "113 年董事會／各委員會績效評估結果是於哪一日期報告至各會議？"),
    (9, "中", "113 年度評鑑平均總分為何？（a）整體董事會；（b）個別董事成員。"),
    (10, "中", "三大功能性委員會（薪資報酬、審計、提名）的平均總分各為多少？"),
    (11, "中", "在兩年度比較中，113 年董事酬金總額為多少？相較 112 年變動多少？"),
    (12, "中", "（副）總經理酬金總額占比自 112→113 年下降的主因為何？"),
    (13, "中", "本公司董事酬金占稅後純益之比例在 112 年與 113 年分別為何？"),
    (14, "中", "（副）總經理酬金占稅後純益之比例在 112 年與 113 年分別為何？"),
    (15, "中", "依公司政策，董事酬勞得自年度盈餘提列之上限比例為多少？"),
    (16, "中", "依公司政策，當年度有盈餘時，員工酬勞應提列之比例為多少？"),
    (17, "中", "113 年度分配予總經理之員工酬勞（股票／現金／合計）各為多少？其占比為多少？"),
    (18, "難", "何種情形需要揭露個別董事酬金資訊？請列舉任兩項條件。"),
    (19, "難", "因設置審計委員會而不適用證券交易法第 14 條之 3 之規定時，實際適用的是哪一條？"),
    (20, "難", "根據提供的數據，驗證總經理 3,640 仟元約等於 113 年個體稅後純益的 0.35%。請寫出計算式（分數）並給出結果。"),
]

# API Configuration
API_BASE_URL = "http://127.0.0.1:5000"  # Adjust if your API runs on different port
CHAT_ENDPOINT = f"{API_BASE_URL}/api/chat/message"
STATUS_ENDPOINTS = [f"{API_BASE_URL}/health", f"{API_BASE_URL}/api/status/health"]

class RetrievalTester:
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
        self.total_api_time = 0.0  # Track total time spent on API calls
        
    def check_api_status(self) -> bool:
        """Check if the API is running (try multiple health endpoints)."""
        for url in STATUS_ENDPOINTS:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except Exception:
                continue
        return False
    
    def send_question(self, question: str, room_id: str | None = None) -> Dict[str, Any]:
        """Send a question to the chat API and get response (auto-create room when room_id=None)."""
        payload = {
            "content": question,
            "user_id": "test_script",
            "room_id": room_id,
            # chat route auto-creates room when None; no streaming in this test
        }
        
        try:
            api_start_time = time.time()  # Start timing API call
            response = requests.post(CHAT_ENDPOINT, json=payload, timeout=180)
            api_end_time = time.time()  # End timing API call
            api_call_time = api_end_time - api_start_time
            self.total_api_time += api_call_time  # Accumulate API time
            
            if response.status_code == 200:
                result = response.json()
                result['api_call_time'] = api_call_time  # Add timing info to result
                return result
            else:
                return {
                    "error": f"API returned status {response.status_code}",
                    "response": response.text,
                    "api_call_time": api_call_time
                }
        except Exception as e:
            return {"error": f"Request failed: {e}", "api_call_time": 0.0}
    
    def evaluate_response_quality(self, question: str, response: str, difficulty: str) -> Dict[str, Any]:
        """Evaluate the quality of a response based on various metrics."""
        evaluation = {
            "response_length": len(response),
            "has_error": "error" in response.lower() or "抱歉" in response or "無法" in response,
            "has_numbers": any(char.isdigit() for char in response),
            "has_percentage": "%" in response,
            "has_chinese_numbers": any(char in "一二三四五六七八九十百千萬億" for char in response),
            "difficulty": difficulty
        }
        
        # Content relevance indicators
        relevance_keywords = {
            "總經理": ["總經理", "經理", "執行長", "CEO"],
            "酬金": ["酬金", "薪資", "報酬", "薪資報酬"],
            "董事": ["董事", "董事會", "獨立董事"],
            "稅後純益": ["稅後純益", "淨利", "純益", "稅後"],
            "比例": ["比例", "百分比", "%", "占比"],
            "年度": ["年度", "年", "113年", "112年"],
            "會議": ["會議", "開會", "召開"],
            "出席": ["出席", "出席率", "委託"],
            "委員會": ["委員會", "薪資報酬委員會", "審計委員會", "提名委員會"],
            "政策": ["政策", "規定", "法規"],
            "員工酬勞": ["員工酬勞", "股票", "現金"],
            "揭露": ["揭露", "公開", "資訊"],
            "證券交易法": ["證券交易法", "第14條", "第14條之3"]
        }
        
        # Check keyword relevance
        question_lower = question.lower()
        response_lower = response.lower()
        
        relevance_score = 0
        total_keywords = 0
        
        for category, keywords in relevance_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                total_keywords += 1
                if any(keyword in response_lower for keyword in keywords):
                    relevance_score += 1
        
        evaluation["keyword_relevance"] = relevance_score / max(total_keywords, 1)
        evaluation["total_keywords"] = total_keywords
        evaluation["matched_keywords"] = relevance_score
        
        # Overall quality score (0-10)
        quality_score = 0
        
        # Base score for having a response
        if not evaluation["has_error"]:
            quality_score += 3
        
        # Score for content length (not too short, not too long)
        if 50 <= evaluation["response_length"] <= 1000:
            quality_score += 2
        elif evaluation["response_length"] > 1000:
            quality_score += 1
        
        # Score for keyword relevance
        quality_score += evaluation["keyword_relevance"] * 3
        
        # Score for having numerical content (important for financial questions)
        if evaluation["has_numbers"] or evaluation["has_percentage"]:
            quality_score += 1
        
        # Bonus for Chinese numbers (common in financial documents)
        if evaluation["has_chinese_numbers"]:
            quality_score += 0.5
        
        evaluation["quality_score"] = min(quality_score, 10)
        
        return evaluation
    
    def run_test(self, question_id: int, difficulty: str, question: str) -> Dict[str, Any]:
        """Run a single test question."""
        print(f"\n🔍 Testing Question {question_id} ({difficulty}): {question}")
        
        # Send question to API (auto-create room)
        response_data = self.send_question(question, room_id=None)
        
        # Extract API call time
        api_call_time = response_data.get('api_call_time', 0.0)
        
        if "error" in response_data:
            result = {
                "question_id": question_id,
                "difficulty": difficulty,
                "question": question,
                "response": f"Error: {response_data['error']}",
                "evaluation": {
                    "quality_score": 0,
                    "has_error": True,
                    "response_length": 0,
                    "keyword_relevance": 0
                },
                "api_call_time": api_call_time,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Extract response text from the actual API structure
            if response_data.get("status") == "success" and "data" in response_data:
                # Get the last AI message from the messages array
                messages = response_data["data"].get("messages", [])
                ai_messages = [msg for msg in messages if msg.get("role") == "ai"]
                
                if ai_messages:
                    # Get the most recent AI response
                    response_text = ai_messages[-1].get("content", "No response received")
                else:
                    response_text = "No AI response found"
            else:
                response_text = response_data.get("response", "No response received")
            
            # Evaluate response quality
            evaluation = self.evaluate_response_quality(question, response_text, difficulty)
            
            result = {
                "question_id": question_id,
                "difficulty": difficulty,
                "question": question,
                "response": response_text,
                "evaluation": evaluation,
                "api_call_time": api_call_time,
                "timestamp": datetime.now().isoformat()
            }
        
        # Print result summary
        score = result["evaluation"]["quality_score"]
        print(f"   📊 Quality Score: {score:.1f}/10")
        print(f"   📝 Response Length: {result['evaluation']['response_length']} chars")
        print(f"   🎯 Keyword Relevance: {result['evaluation']['keyword_relevance']:.2f}")
        print(f"   ⏱️  API Call Time: {api_call_time:.2f}s")
        
        return result
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all test questions."""
        print("🚀 Starting Retrieval Quality Test")
        print("=" * 50)
        
        # Check API status
        if not self.check_api_status():
            print("❌ API is not accessible. Please make sure the API is running.")
            return []
        
        print("✅ API is accessible")
        
        self.start_time = time.time()
        
        for question_id, difficulty, question in questions:
            result = self.run_test(question_id, difficulty, question)
            self.results.append(result)
            
            # Small delay between requests
            time.sleep(1)
        
        self.end_time = time.time()
        
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of test results."""
        if not self.results:
            return {"error": "No results to summarize"}
        
        total_questions = len(self.results)
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate overall statistics
        quality_scores = [r["evaluation"]["quality_score"] for r in self.results]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # API call time statistics
        api_call_times = [r.get("api_call_time", 0.0) for r in self.results]
        total_api_time = sum(api_call_times)
        avg_api_call_time = total_api_time / len(api_call_times) if api_call_times else 0
        min_api_call_time = min(api_call_times) if api_call_times else 0
        max_api_call_time = max(api_call_times) if api_call_times else 0
        
        # Calculate overhead time (total time - API time)
        overhead_time = total_time - total_api_time
        overhead_percentage = (overhead_time / total_time * 100) if total_time > 0 else 0
        
        # Statistics by difficulty
        difficulty_stats = {}
        for difficulty in ["易", "中", "難"]:
            diff_results = [r for r in self.results if r["difficulty"] == difficulty]
            if diff_results:
                diff_scores = [r["evaluation"]["quality_score"] for r in diff_results]
                diff_api_times = [r.get("api_call_time", 0.0) for r in diff_results]
                difficulty_stats[difficulty] = {
                    "count": len(diff_results),
                    "avg_score": sum(diff_scores) / len(diff_scores),
                    "min_score": min(diff_scores),
                    "max_score": max(diff_scores),
                    "avg_api_call_time": sum(diff_api_times) / len(diff_api_times),
                    "total_api_time": sum(diff_api_times)
                }
        
        # Error analysis
        error_count = sum(1 for r in self.results if r["evaluation"]["has_error"])
        success_rate = (total_questions - error_count) / total_questions * 100
        
        # Response length analysis
        response_lengths = [r["evaluation"]["response_length"] for r in self.results]
        avg_length = sum(response_lengths) / len(response_lengths)
        
        # Keyword relevance analysis
        keyword_relevance = [r["evaluation"]["keyword_relevance"] for r in self.results]
        avg_relevance = sum(keyword_relevance) / len(keyword_relevance)
        
        summary = {
            "test_summary": {
                "total_questions": total_questions,
                "total_time_seconds": total_time,
                "avg_time_per_question": total_time / total_questions if total_questions > 0 else 0,
                "success_rate_percent": success_rate,
                "error_count": error_count
            },
            "api_timing": {
                "total_api_time_seconds": total_api_time,
                "avg_api_call_time_seconds": avg_api_call_time,
                "min_api_call_time_seconds": min_api_call_time,
                "max_api_call_time_seconds": max_api_call_time,
                "overhead_time_seconds": overhead_time,
                "overhead_percentage": overhead_percentage,
                "api_efficiency_percentage": (total_api_time / total_time * 100) if total_time > 0 else 0
            },
            "quality_metrics": {
                "overall_avg_score": avg_quality,
                "min_score": min(quality_scores),
                "max_score": max(quality_scores),
                "avg_response_length": avg_length,
                "avg_keyword_relevance": avg_relevance
            },
            "difficulty_breakdown": difficulty_stats,
            "top_performers": sorted(self.results, key=lambda x: x["evaluation"]["quality_score"], reverse=True)[:5],
            "bottom_performers": sorted(self.results, key=lambda x: x["evaluation"]["quality_score"])[:5],
            "fastest_responses": sorted(self.results, key=lambda x: x.get("api_call_time", 0.0))[:5],
            "slowest_responses": sorted(self.results, key=lambda x: x.get("api_call_time", 0.0), reverse=True)[:5]
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary of test results."""
        print("\n" + "=" * 60)
        print("📊 RETRIEVAL QUALITY TEST SUMMARY")
        print("=" * 60)
        
        # Test overview
        test_summary = summary["test_summary"]
        print(f"\n🔍 Test Overview:")
        print(f"   • Total Questions: {test_summary['total_questions']}")
        print(f"   • Total Time: {test_summary['total_time_seconds']:.1f} seconds")
        print(f"   • Avg Time per Question: {test_summary['avg_time_per_question']:.1f} seconds")
        print(f"   • Success Rate: {test_summary['success_rate_percent']:.1f}%")
        print(f"   • Error Count: {test_summary['error_count']}")
        
        # API timing analysis
        api_timing = summary["api_timing"]
        print(f"\n⏱️  API Timing Analysis:")
        print(f"   • Total API Time: {api_timing['total_api_time_seconds']:.1f} seconds")
        print(f"   • Avg API Call Time: {api_timing['avg_api_call_time_seconds']:.2f} seconds")
        print(f"   • API Call Range: {api_timing['min_api_call_time_seconds']:.2f}s - {api_timing['max_api_call_time_seconds']:.2f}s")
        print(f"   • Overhead Time: {api_timing['overhead_time_seconds']:.1f} seconds ({api_timing['overhead_percentage']:.1f}%)")
        print(f"   • API Efficiency: {api_timing['api_efficiency_percentage']:.1f}%")
        
        # Quality metrics
        quality_metrics = summary["quality_metrics"]
        print(f"\n📈 Quality Metrics:")
        print(f"   • Overall Avg Score: {quality_metrics['overall_avg_score']:.2f}/10")
        print(f"   • Score Range: {quality_metrics['min_score']:.1f} - {quality_metrics['max_score']:.1f}")
        print(f"   • Avg Response Length: {quality_metrics['avg_response_length']:.0f} characters")
        print(f"   • Avg Keyword Relevance: {quality_metrics['avg_keyword_relevance']:.2f}")
        
        # Difficulty breakdown
        print(f"\n📊 Performance by Difficulty:")
        for difficulty, stats in summary["difficulty_breakdown"].items():
            print(f"   • {difficulty} ({stats['count']} questions):")
            print(f"     - Avg Score: {stats['avg_score']:.2f}/10")
            print(f"     - Score Range: {stats['min_score']:.1f} - {stats['max_score']:.1f}")
            print(f"     - Avg API Time: {stats['avg_api_call_time']:.2f}s")
            print(f"     - Total API Time: {stats['total_api_time']:.1f}s")
        
        # Top performers
        print(f"\n🏆 Top 5 Performers:")
        for i, result in enumerate(summary["top_performers"], 1):
            score = result["evaluation"]["quality_score"]
            api_time = result.get("api_call_time", 0.0)
            print(f"   {i}. Q{result['question_id']} ({result['difficulty']}): {score:.1f}/10 ({api_time:.2f}s)")
            print(f"      \"{result['question'][:50]}...\"")
        
        # Bottom performers
        print(f"\n⚠️ Bottom 5 Performers:")
        for i, result in enumerate(summary["bottom_performers"], 1):
            score = result["evaluation"]["quality_score"]
            api_time = result.get("api_call_time", 0.0)
            print(f"   {i}. Q{result['question_id']} ({result['difficulty']}): {score:.1f}/10 ({api_time:.2f}s)")
            print(f"      \"{result['question'][:50]}...\"")
        
        # Fastest responses
        print(f"\n⚡ Fastest 5 Responses:")
        for i, result in enumerate(summary["fastest_responses"], 1):
            api_time = result.get("api_call_time", 0.0)
            score = result["evaluation"]["quality_score"]
            print(f"   {i}. Q{result['question_id']} ({result['difficulty']}): {api_time:.2f}s (Score: {score:.1f}/10)")
            print(f"      \"{result['question'][:50]}...\"")
        
        # Slowest responses
        print(f"\n🐌 Slowest 5 Responses:")
        for i, result in enumerate(summary["slowest_responses"], 1):
            api_time = result.get("api_call_time", 0.0)
            score = result["evaluation"]["quality_score"]
            print(f"   {i}. Q{result['question_id']} ({result['difficulty']}): {api_time:.2f}s (Score: {score:.1f}/10)")
            print(f"      \"{result['question'][:50]}...\"")
    
    def save_results(self, filename: str = None):
        """Save test results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retrieval_test_results_{timestamp}.json"
        
        data = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(self.results),
                "api_base_url": API_BASE_URL
            },
            "results": self.results,
            "summary": self.generate_summary()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main function to run the retrieval quality test."""
    tester = RetrievalTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    if results:
        # Generate and print summary
        summary = tester.generate_summary()
        tester.print_summary(summary)
        
        # Save results
        tester.save_results()
    else:
        print("❌ No test results generated. Please check API connectivity.")

if __name__ == "__main__":
    main()
