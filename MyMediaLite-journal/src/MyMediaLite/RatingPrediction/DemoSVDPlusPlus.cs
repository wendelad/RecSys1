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
using System.IO;
using System.Linq;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MyMediaLite.IO;

namespace MyMediaLite.RatingPrediction
{
	/// <summary>
	/// Demo SVD plus plus.
	/// </summary>
	/// <exception cref='IOException'>
	/// Is thrown when an I/O operation fails.
	/// </exception>
	/// <exception cref='Exception'>
	/// Represents errors that occur during application execution.
	/// </exception>
	public class DemoSVDPlusPlus : DemoSVD
	{
		public DemoSVDPlusPlus () : base()
		{
		}
		
		/// <summary>
		/// Iterate once over rating data and adjust corresponding factors (stochastic gradient descent)
		/// </summary>
		/// <param name='rating_indices'>
		/// a list of indices pointing to the ratings to iterate over
		/// </param>
		/// <param name='update_user'>
		/// true if user factors to be updated
		/// </param>
		/// <param name='update_item'>
		/// true if item factors to be updated
		/// </param>
		protected override void Iterate(IList<int> rating_indices, bool update_user, bool update_item)
		{
			user_factors = null; // delete old user factors
			float reg = Regularization; // to limit property accesses			
			
			foreach (int index in rating_indices)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];

				float prediction = global_bias + user_bias[u] + item_bias[i];

				if(u < UserAttributes.NumberOfRows)
				{
					IList<int> attribute_list = UserAttributes.GetEntriesByRow(u);
					if(attribute_list.Count > 0)
					{
						float sum = 0;
						float second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += main_demo[attribute_id];
						}
						prediction += sum / second_norm_denominator;
					}
				}

				for(int d = 0; d < AdditionalUserAttributes.Count; d++)
				{
					if(u < AdditionalUserAttributes[d].NumberOfRows)
					{
						IList<int> attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(u);
						if(attribute_list.Count > 0)
						{
							float sum = 0;
							float second_norm_denominator = attribute_list.Count;
							foreach(int attribute_id in attribute_list) 
							{
								sum += second_demo[d][attribute_id];
							}
							prediction += sum / second_norm_denominator;
						}
					}	
				}

				if(i < ItemAttributes.NumberOfRows)
				{
					IList<int> attribute_list = ItemAttributes.GetEntriesByRow(i);
					if(attribute_list.Count > 0)
					{
						float sum = 0;
						float second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += main_metadata[attribute_id];
						}
						prediction += sum / second_norm_denominator;
					}
				}
				
				for(int g = 0; g < AdditionalItemAttributes.Count; g++)
				{
					if(i < AdditionalItemAttributes[g].NumberOfRows)
					{
						IList<int> attribute_list = AdditionalItemAttributes[g].GetEntriesByRow(i);
						if(attribute_list.Count > 0)
						{
							float sum = 0;
							float second_norm_denominator = attribute_list.Count;
							foreach(int attribute_id in attribute_list) 
							{
								sum += second_metadata[g][attribute_id];
							}
							prediction += sum / second_norm_denominator;
						}
					}
				}

				if (u < UserAttributes.NumberOfRows && i < ItemAttributes.NumberOfRows)
				{
					IList<int> item_attribute_list = ItemAttributes.GetEntriesByRow(i);
					float item_norm_denominator = item_attribute_list.Count;
					
					IList<int> user_attribute_list = UserAttributes.GetEntriesByRow(u);
					float user_norm_denominator = user_attribute_list.Count;
					
					float demo_spec = 0;
					float sum = 0;
					foreach(int u_att in user_attribute_list)
					{
						foreach(int i_att in item_attribute_list)
						{
							sum += h[0][u_att, i_att];
						}
					}				
					demo_spec += sum / user_norm_denominator;
					
					for(int d = 0; d < AdditionalUserAttributes.Count; d++)
					{
						user_attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(u);
						user_norm_denominator = user_attribute_list.Count;
						sum = 0;
						foreach(int u_att in user_attribute_list)
						{
							foreach(int i_att in item_attribute_list)
							{
								sum += h[d + 1][u_att, i_att];
							}
						}				
						demo_spec += sum / user_norm_denominator;
					}
					
					prediction += demo_spec / item_norm_denominator;
				}
				
				var p_plus_y_sum_vector = y.SumOfRows(items_rated_by_user[u]);
				double norm_denominator = Math.Sqrt(items_rated_by_user[u].Length);
				for (int f = 0; f < p_plus_y_sum_vector.Count; f++)
					p_plus_y_sum_vector[f] = (float) (p_plus_y_sum_vector[f] / norm_denominator + p[u, f]);

				prediction += DataType.MatrixExtensions.RowScalarProduct(item_factors, i, p_plus_y_sum_vector);
				

				float err = ratings[index] - prediction;
				
				float user_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByUser[u])) : reg;
				float item_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByItem[i])) : reg;

				// adjust biases
				if (update_user)
					user_bias[u] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * user_reg_weight * user_bias[u]);
				if (update_item)
					item_bias[i] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * item_reg_weight * item_bias[i]);
				
				// adjust demo global attributes
				if(u < UserAttributes.NumberOfRows)
				{
					IList<int> attribute_list = UserAttributes.GetEntriesByRow(u);
					if(attribute_list.Count > 0)
					{
						foreach (int attribute_id in attribute_list)
						{							
							main_demo[attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * main_demo[attribute_id]);
						}
					}
				}
				
				for(int d = 0; d < AdditionalUserAttributes.Count; d++)
				{
					if(u < AdditionalUserAttributes[d].NumberOfRows)
					{
						IList<int> attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(u);
						if(attribute_list.Count > 0)
						{
							foreach (int attribute_id in attribute_list)
							{
								second_demo[d][attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * second_demo[d][attribute_id]);								
							}
						}
					}	
				}
				
				// adjust metadata global attributes
				// adjust item attributes
				if(i < ItemAttributes.NumberOfRows)
				{
					IList<int> attribute_list = ItemAttributes.GetEntriesByRow(i);
					if(attribute_list.Count > 0)
					{
						foreach (int attribute_id in attribute_list)
						{							
							main_metadata[attribute_id] += BiasLearnRate * current_learnrate * ((float)err - BiasReg * Regularization * main_metadata[attribute_id]);
						}
					}
				}
				
				for(int g = 0; g < AdditionalItemAttributes.Count; g++)
				{
					if(i < AdditionalItemAttributes[g].NumberOfRows)
					{
						IList<int> attribute_list = AdditionalItemAttributes[g].GetEntriesByRow(i);
						if(attribute_list.Count > 0)
						{
							foreach (int attribute_id in attribute_list)
							{
								second_metadata[g][attribute_id] += BiasLearnRate * current_learnrate * ((float)err - BiasReg * Regularization * second_metadata[g][attribute_id]);								
							}
						}
					}	
				}
				
				// adjust demo specific attributes
				if(u < UserAttributes.NumberOfRows && i < ItemAttributes.NumberOfRows)
				{
					IList<int> item_attribute_list = ItemAttributes.GetEntriesByRow(i);
					float item_norm_denominator = item_attribute_list.Count;
					
					IList<int> user_attribute_list = UserAttributes.GetEntriesByRow(u);
					float user_norm = 1 / user_attribute_list.Count;				
					
					float norm_error = err / item_norm_denominator;
					
					foreach(int u_att in user_attribute_list)
					{
						foreach(int i_att in item_attribute_list)
						{
							h[0][u_att, i_att] += current_learnrate * (norm_error * user_norm - Regularization * h[0][u_att, i_att]);
						}
					}								
					
					for(int d = 0; d < AdditionalUserAttributes.Count; d++)
					{
						user_attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(u);
						user_norm = 1 / user_attribute_list.Count;
						
						foreach(int u_att in user_attribute_list)
						{
							foreach(int i_att in item_attribute_list)
							{
								h[d + 1][u_att, i_att] += current_learnrate * (norm_error * user_norm - Regularization * h[d + 1][u_att, i_att]);;
							}
						}									
					}
				}
				
				// adjust factors
				double normalized_error = err / norm_denominator;
				for (int f = 0; f < NumFactors; f++)
				{
					float i_f = item_factors[i, f];

					// if necessary, compute and apply updates
					if (update_user)
					{
						double delta_u = err * i_f - user_reg_weight * p[u, f];
						p.Inc(u, f, current_learnrate * delta_u);
					}
					if (update_item)
					{
						double delta_i = err * p_plus_y_sum_vector[f] - item_reg_weight * i_f;
						item_factors.Inc(i, f, current_learnrate * delta_i);
						double common_update = normalized_error * i_f;
						foreach (int other_item_id in items_rated_by_user[u])
						{
							double delta_oi = common_update - y_reg[other_item_id] * y[other_item_id, f];
							y.Inc(other_item_id, f, current_learnrate * delta_oi);
						}
					}

				}
			}

			UpdateLearnRate();
		}
		
		/// <summary>
		/// Predict the specified user_id, item_id and bound.
		/// </summary>
		/// <param name='user_id'>
		/// User_id.
		/// </param>
		/// <param name='item_id'>
		/// Item_id.
		/// </param>
		public override float Predict(int user_id, int item_id)
		{
			double result = global_bias;

			if (user_factors == null)
				PrecomputeUserFactors();
			
			if (user_id < user_bias.Length)
				result += user_bias[user_id];
			if (item_id < item_bias.Length)
				result += item_bias[item_id];
			
			if(user_id < UserAttributes.NumberOfRows)
			{
				IList<int> attribute_list = UserAttributes.GetEntriesByRow(user_id);
				if(attribute_list.Count > 0)
				{
					double sum = 0;
					double second_norm_denominator = attribute_list.Count;
					foreach(int attribute_id in attribute_list) 
					{
						sum += main_demo[attribute_id];
					}
					result += sum / second_norm_denominator;
				}
			}

			for(int d = 0; d < AdditionalUserAttributes.Count; d++)
			{
				if(user_id < AdditionalUserAttributes[d].NumberOfRows)
				{
					IList<int> attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(user_id);
					if(attribute_list.Count > 0)
					{
						double sum = 0;
						double second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += second_demo[d][attribute_id];
						}
						result += sum / second_norm_denominator;
					}
				}	
			}

			if(item_id < ItemAttributes.NumberOfRows)
			{
				IList<int> attribute_list = ItemAttributes.GetEntriesByRow(item_id);
				if(attribute_list.Count > 0)
				{
					double sum = 0;
					double second_norm_denominator = attribute_list.Count;
					foreach(int attribute_id in attribute_list) 
					{
						sum += main_metadata[attribute_id];
					}
					result += sum / second_norm_denominator;
				}
			}
			
			for(int g = 0; g < AdditionalItemAttributes.Count; g++)
			{
				if(item_id < AdditionalItemAttributes[g].NumberOfRows)
				{
					IList<int> attribute_list = AdditionalItemAttributes[g].GetEntriesByRow(item_id);
					if(attribute_list.Count > 0)
					{
						double sum = 0;
						double second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += second_metadata[g][attribute_id];
						}
						result += sum / second_norm_denominator;
					}
				}	
			}
						
			if (user_id < UserAttributes.NumberOfRows && item_id < ItemAttributes.NumberOfRows) {
				IList<int> item_attribute_list = ItemAttributes.GetEntriesByRow (item_id);
				double item_norm_denominator = item_attribute_list.Count;
				
				IList<int> user_attribute_list = UserAttributes.GetEntriesByRow (user_id);
				float user_norm_denominator = user_attribute_list.Count;
				
				float demo_spec = 0;
				float sum = 0;
				foreach (int u_att in user_attribute_list) {
					foreach (int i_att in item_attribute_list) {
						sum += h [0] [u_att, i_att];
					}
				}
				demo_spec += sum / user_norm_denominator;

				for (int d = 0; d < AdditionalUserAttributes.Count; d++) {
					user_attribute_list = AdditionalUserAttributes [d].GetEntriesByRow (user_id);
					user_norm_denominator = user_attribute_list.Count;
					sum = 0;
					foreach (int u_att in user_attribute_list) {
						foreach (int i_att in item_attribute_list) {
							sum += h [d + 1] [u_att, i_att];
						}
					}				
					demo_spec += sum / user_norm_denominator;
				}
				
				result += demo_spec / item_norm_denominator;
			}

			if (user_id <= MaxUserID && item_id <= MaxItemID)
				result += DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);

			if (result > MaxRating)
				return MaxRating;
			if (result < MinRating)
				return MinRating;

			return (float)result;
		}

		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"{0} num_factors={1} regularization={2} bias_reg={3} frequency_regularization={4} learn_rate={5} bias_learn_rate={6} decay={7} num_iter={8}",
				this.GetType().Name, NumFactors, Regularization, BiasReg, FrequencyRegularization, LearnRate, BiasLearnRate, Decay, NumIter);
		}
	}
}
