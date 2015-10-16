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
	/// Demo user item att.
	/// </summary>
	public class DemoUserItemAtt : DemoUserBaseline
	{
		///
		protected Matrix<float>[] h;
		
		
		public DemoUserItemAtt () : base()
		{			
		}
		
		///
		protected internal override void InitModel()
		{
			base.InitModel();
			
			h = new Matrix<float>[AdditionalUserAttributes.Count + 1];
			h[0] = new Matrix<float>(UserAttributes.NumberOfColumns, ItemAttributes.NumberOfColumns);
			h[0].InitNormal(InitMean, InitStdDev);
			for(int d = 0; d < AdditionalUserAttributes.Count; d++)
			{
				h[d + 1] = new Matrix<float>(AdditionalUserAttributes[d].NumberOfColumns, ItemAttributes.NumberOfColumns);
				h[d + 1].InitNormal(InitMean, InitStdDev);
			}
		}
		
		///
		protected override void Iterate(IList<int> rating_indices, bool update_user, bool update_item)
		{
			float reg = Regularization; // to limit property accesses			
			
			foreach (int index in rating_indices)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];

				float prediction = Predict(u, i, false);
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

				// adjust item attributes
				if(i < ItemAttributes.NumberOfRows)
				{
					IList<int> attribute_list = ItemAttributes.GetEntriesByRow(i);
					if(attribute_list.Count > 0)
					{
						foreach (int attribute_id in attribute_list)
						{							
							main_metadata[attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * main_metadata[attribute_id]);
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
								second_metadata[g][attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * second_metadata[g][attribute_id]);								
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
			}

			UpdateLearnRate();
		}

		///
		protected override float Predict(int user_id, int item_id, bool bound)
		{						
			double result = base.Predict(user_id, item_id, false);
						
			if (user_id < UserAttributes.NumberOfRows && item_id < ItemAttributes.NumberOfRows)
			{
				IList<int> item_attribute_list = ItemAttributes.GetEntriesByRow(item_id);
				double item_norm_denominator = item_attribute_list.Count;
				
				IList<int> user_attribute_list = UserAttributes.GetEntriesByRow(user_id);
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
					user_attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(user_id);
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
				
				result += demo_spec / item_norm_denominator;
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

			if (bound)
			{
				if (result > MaxRating)
					return MaxRating;
				if (result < MinRating)
					return MinRating;
			}
			return (float)result;
		}

		/// <summary>Predict the rating of a given user for a given item</summary>		
		/// <param name="user_id">the user ID</param>
		/// <param name="item_id">the item ID</param>
		/// <returns>the predicted rating</returns>
		public override float Predict(int user_id, int item_id)
		{
			return Predict(user_id, item_id, true);
		}
		
///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"{0} bias_reg={1} reg_u={2} reg_i={3} frequency_regularization={4} learn_rate={5} bias_learn_rate={6} learn_rate_decay={7} num_iter={8}",
				this.GetType().Name, BiasReg, RegU, RegI, FrequencyRegularization, LearnRate, BiasLearnRate, Decay, NumIter);
		}
	}
}

