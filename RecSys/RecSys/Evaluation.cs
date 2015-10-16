using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MyMediaLite.Eval;
using MyMediaLite.IO;
using MyMediaLite.ItemRecommendation;
using MyMediaLite.Eval.Measures;


namespace RecSys
{
    class Evaluation
    {
        protected List<IRecommender> m_recommenders;

        protected Ensemble m_ensemble;

        protected List<IPosOnlyFeedback> m_test_probe_data;
        protected List<IPosOnlyFeedback> m_training_probe_data;

        public Evaluation(List<IRecommender> recommenders, List<IPosOnlyFeedback> test_probe_data,
            List<IPosOnlyFeedback> training_probe_data)
        {
            m_recommenders = recommenders;
            m_test_probe_data = test_probe_data;
            m_training_probe_data = training_probe_data;
            m_ensemble = new Ensemble(m_recommenders);
            
        }


        static public ICollection<string> Measures
        {
            get
            {
                string[] measures = { "AUC", "prec@5", "prec@10", "MAP", "recall@5", "recall@10", "NDCG", "MRR" };
                return new HashSet<string>(measures);
            }
        }

        public List<ItemRecommendationEvaluationResults> Evaluate(List<IPosOnlyFeedback> test_data, 
            List<IPosOnlyFeedback> training_data, List<IList<int>> test_users, List<IMapping> user_mapping,
            List<IMapping> item_mapping,

            int n = -1)
        {
            List<IList<int>> candidate_items = new List<IList<int>>();
            List<RepeatedEvents> repeated_events = new List<RepeatedEvents>();
            List<IBooleanMatrix> training_user_matrix = new List<IBooleanMatrix>();
            List<IBooleanMatrix> test_user_matrix = new List<IBooleanMatrix>();


            for (int i = 0; i < m_recommenders.Count; i++)
            {

                candidate_items.Add(new List<int>(test_data[i].AllItems.Union(training_data[i].AllItems)));
                repeated_events.Add(RepeatedEvents.No);


                if (candidate_items[i] == null)
                    throw new ArgumentNullException("candidate_items");
                if (test_users[i] == null)
                    test_users[i] = test_data[i].AllUsers;

                training_user_matrix.Add(training_data[i].UserMatrix);
                test_user_matrix.Add(test_data[i].UserMatrix);
            }
            int num_users = 0;
            var result = new List<ItemRecommendationEvaluationResults>();

            for (int i = 0; i < m_recommenders.Count + 3; i++) // +Ensemble +GA
            {
                result.Add(new ItemRecommendationEvaluationResults());
            }

            // make sure that the user matrix is completely initialized before entering parallel code






            foreach (int user_id in test_users[0])
            {

                string original = user_mapping[0].ToOriginalID(user_id);


                List<IList<Tuple<int, float>>> list_of_predictions = new List<IList<Tuple<int, float>>>();

                HashSet<int> correct_items = new HashSet<int>();

                List<HashSet<int>> ignore_items_for_this_user = new List<HashSet<int>>();

                List<int> num_candidates_for_this_user = new List<int>();


                correct_items = new HashSet<int>(test_user_matrix[0][user_id]);
                correct_items.IntersectWith(candidate_items[0]);


                for (int i = 0; i < m_recommenders.Count; i++)
                {

                    int internalId = user_mapping[i].ToInternalID(original);


                    ignore_items_for_this_user.Add(new HashSet<int>(training_user_matrix[i][internalId]));



                    /* if (correct_items[i].Count == 0)
                         continue;
                     */

                    ignore_items_for_this_user[i].IntersectWith(candidate_items[i]);
                    num_candidates_for_this_user.Add(candidate_items[i].Count - ignore_items_for_this_user[i].Count);
                    /*if (correct_items[i].Count == num_candidates_for_this_user[i])
                        continue;
                    */


                    //Recomenda


                    var listaRecomendacao = m_recommenders[i].Recommend(user_id, candidate_items: candidate_items[i], n: n, ignore_items: ignore_items_for_this_user[i]);
                    for (int j = 0; j < listaRecomendacao.Count; j++)
                    {
                        string idOriginal = item_mapping[i].ToOriginalID(listaRecomendacao[j].Item1);
                        int idMappingZero = item_mapping[0].ToInternalID(idOriginal);


                        Tuple<int, float> tupla = new Tuple<int, float>(idMappingZero, listaRecomendacao[j].Item2);

                        listaRecomendacao[j] = tupla;
                    }

                    list_of_predictions.Add(listaRecomendacao);


                }




                //}

                //Nova
                //var prediction = Ensenble(list_of_predictions);
                //var prediction_list = (from t in prediction select t.Key).ToArray();






                for (int i = 0; i < m_recommenders.Count + 3; i++) // +Ensemble +GA
                {

                    int best = m_ensemble.best_alg[original];

                    IList<int> prediction_list = null;
                    int prediction_count = 0;


                    if (i == list_of_predictions.Count)//Best of all
                    {
                        var prediction = list_of_predictions[best];
                        prediction_list = (from t in prediction select t.Item1).ToArray();
                        prediction_count = prediction.Count;
                    }
                    else if (i == list_of_predictions.Count + 1)//emsemble
                    {
                        var prediction_ensemble = m_ensemble.Ensenble(list_of_predictions);

                        prediction_list = (from t in prediction_ensemble select t.Key).ToArray();
                        prediction_count = prediction_ensemble.Count;
                    }
                    else if (i == list_of_predictions.Count + 2)//GA
                    {
                        //Set global so Fitness itens can see.
                        m_ensemble.list_prediction_probes = list_of_predictions;
                        m_ensemble.correct_items_global = correct_items;

                        var prediction_ensemble = m_ensemble.EnsenblePeso(m_ensemble.ga_weights[original].ToArray());

                        prediction_list = (from t in prediction_ensemble select t.Key).ToArray();
                        prediction_count = prediction_ensemble.Count;
                    }
                    else
                    {
                        var prediction = list_of_predictions[i];
                        prediction_list = (from t in prediction select t.Item1).ToArray();
                        prediction_count = prediction.Count;
                    }




                    int num_dropped_items = num_candidates_for_this_user[0] - prediction_count;
                    double auc = AUC.Compute(prediction_list, correct_items, num_dropped_items);
                    double map = PrecisionAndRecall.AP(prediction_list, correct_items);
                    double ndcg = NDCG.Compute(prediction_list, correct_items);
                    double rr = ReciprocalRank.Compute(prediction_list, correct_items);
                    var positions = new int[] { 5, 10 };
                    var prec = PrecisionAndRecall.PrecisionAt(prediction_list, correct_items, positions);
                    var recall = PrecisionAndRecall.RecallAt(prediction_list, correct_items, positions);

                    // thread-safe incrementing

                    num_users++;
                    result[i]["AUC"] += (float)auc;
                    result[i]["MAP"] += (float)map;
                    result[i]["NDCG"] += (float)ndcg;
                    result[i]["MRR"] += (float)rr;
                    result[i]["prec@5"] += (float)prec[5];
                    result[i]["prec@10"] += (float)prec[10];
                    result[i]["recall@5"] += (float)recall[5];
                    result[i]["recall@10"] += (float)recall[10];


                }





                if (num_users % 1000 == 0)
                    Console.Error.Write(".");
                if (num_users % 60000 == 0)
                    Console.Error.WriteLine();

            }


            num_users /= m_recommenders.Count + 3;

            for (int i = 0; i < m_recommenders.Count + 3; i++) // +Ensemble +GA
            {
                foreach (string measure in Measures)
                    result[i][measure] /= num_users;
                result[i]["num_users"] = num_users;
                result[i]["num_lists"] = num_users;
                result[i]["num_items"] = candidate_items.Count;
            }

            return result;
        }


        public void EvaluateProbe(List<IList<int>> test_users, List<IMapping> user_mapping,
            List<IMapping> item_mapping, int n = -1)
        {
            m_ensemble.EvaluateProbe(m_test_probe_data, m_training_probe_data, test_users, user_mapping, item_mapping);
        }




    }
}
