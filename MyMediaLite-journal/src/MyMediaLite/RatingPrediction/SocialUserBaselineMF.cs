// Copyright (C) 2012 Zeno Gantner
// 
// This file is part of MyMediaLite.
// 
// MyMediaLite is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// MyMediaLite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
//  You should have received a copy of the GNU General Public License
//  along with MyMediaLite.  If not, see <http://www.gnu.org/licenses/>.
// 
using System;
using System.Collections.Generic;
using System.Globalization;
using MyMediaLite.Data;
using MyMediaLite.Correlation;
using MyMediaLite.DataType;
using MyMediaLite.Taxonomy;
using System.Linq;

namespace MyMediaLite.RatingPrediction
{
	/// <summary>
	/// Social-based.
	/// </summary>
	public class SocialUserBaselineMF : SocialUserBaseline
	{
		public SocialUserBaselineMF () : base()
		{		
		}

		///
		protected override void Iterate(IList<int> rating_indices, bool update_user, bool update_item)
		{
			SetupLoss();		
			
			foreach (int index in rating_indices)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];

				double score = Predict (u, i, false);
				double sig_score = 1 / (1 + Math.Exp(-score));

				double prediction = min_rating + sig_score * rating_range_size;
				double err = ratings[index] - prediction;

				float user_reg_weight = FrequencyRegularization ? (float) (RegU / Math.Sqrt(ratings.CountByUser[u])) : RegU;
				float item_reg_weight = FrequencyRegularization ? (float) (RegI / Math.Sqrt(ratings.CountByItem[i])) : RegI;

				// adjust biases
				if (update_user)
					user_bias[u] += BiasLearnRate * current_learnrate * (float)(err - BiasReg * user_reg_weight * user_bias[u]);
				if (update_item)
					item_bias[i] += BiasLearnRate * current_learnrate * (float)(err - BiasReg * item_reg_weight * item_bias[i]);

				// adjust latent factors
				for (int f = 0; f < NumFactors; f++)
				{
					double u_f = user_factors[u, f];
					double i_f = item_factors[i, f];

					if (update_user)
					{
						double delta_u = err * i_f - user_reg_weight * u_f;
						user_factors.Inc(u, f, current_learnrate * delta_u);
					}
					if (update_item)
					{
						double delta_i = err * u_f - item_reg_weight * i_f;
						item_factors.Inc(i, f, current_learnrate * delta_i);
					}
				}

				// adjust groups
				if(u < user_connections.NumberOfRows)
				{
					IList<int> connection_list = user_connections.GetEntriesByRow(u);
					if(connection_list.Count > 0)
					{
						foreach (int connection_id in connection_list)
						{							
							group[connection_id] += BiasLearnRate * current_learnrate * (float)(err - BiasReg * Regularization * group[connection_id]);
						}
					}
				}
			}

			UpdateLearnRate();
		}
		
		///
		protected override float Predict(int user_id, int item_id, bool bound)
		{						
			double result = global_bias;

			if (user_id < user_bias.Length)
				result += user_bias[user_id];
			if (item_id < item_bias.Length)
				result += item_bias[item_id];

			if(user_id < user_connections.NumberOfRows)
			{
				IList<int> connection_list = user_connections.GetEntriesByRow(user_id);
				if(connection_list != null && connection_list.Count > 0)
				{
					double sum = 0;
					double second_norm_denominator = connection_list.Count;
					foreach(int connection_id in connection_list) 
					{
						sum += group[connection_id];
					}
					result += sum / second_norm_denominator;
				}
			}

			result += DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);

			if (bound)
			{
				if (result > MaxRating)
					return MaxRating;
				if (result < MinRating)
					return MinRating;
			}
			return (float)result;
		}
	
		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"{0} num_factors={1} bias_reg={2} reg_u={3} reg_i={4} frequency_regularization={5} learn_rate={6} bias_learn_rate={7} learn_rate_decay={8} num_iter={9}",
				this.GetType().Name, NumFactors, BiasReg, RegU, RegI, FrequencyRegularization, LearnRate, BiasLearnRate, Decay, NumIter);
		}
	}
}