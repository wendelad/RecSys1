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
	/// Demographic-based SVD.
	/// </summary>
	public class DemoSVD : GSVDPlusPlus, IUserAttributeAwareRecommender
	{
		///
		public IBooleanMatrix UserAttributes
		{
			get { return this.user_attributes; }
			set {
				this.user_attributes = value;
				this.NumUserAttributes = user_attributes.NumberOfColumns;
				this.MaxUserID = Math.Max(MaxUserID, user_attributes.NumberOfRows - 1);
			}
		}
		private IBooleanMatrix user_attributes;
		
		///
		public List<IBooleanMatrix> AdditionalUserAttributes
		{
			get { return this.additional_user_attributes; }
			set {
				this.additional_user_attributes = value;
			}
		}
		private List<IBooleanMatrix> additional_user_attributes;
		
		///
		public int NumUserAttributes { get; private set; }
		
		/// <summary>Main demographic biases</summary>
		protected float[] main_demo;
		
		/// <summary>Secondary biases</summary>
		protected List<float[]> second_demo;
		
		/// <summary>Main metadata biases</summary>
		protected float[] main_metadata;
		
		/// <summary>Secondary biases</summary>
		protected List<float[]> second_metadata;
		
		///
		protected Matrix<float>[] h;
		
		/// <summary>
		/// Initializes a new instance of the <see cref="MyMediaLite.RatingPrediction.DemoSVD"/> class.
		/// </summary>
		public DemoSVD () : base()
		{
		}
		
		///
		protected internal override void InitModel()
		{
			base.InitModel();
			
			main_demo = new float[user_attributes.NumberOfColumns];
			second_demo = new List<float[]>(additional_user_attributes.Count);
			for(int d = 0; d < additional_user_attributes.Count; d++)
			{
				float[] element = new float[additional_user_attributes[d].NumberOfColumns];			
				second_demo.Add(element);
			}	
			
			main_metadata = new float[item_attributes.NumberOfColumns];
			second_metadata = new List<float[]>(AdditionalItemAttributes.Count);
			for(int g = 0; g < AdditionalItemAttributes.Count; g++)
			{
				float[] element = new float[AdditionalItemAttributes[g].NumberOfColumns];			
				second_metadata.Add(element);
			}
			
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

				double prediction = global_bias + user_bias[u] + item_bias[i];
				
				if(u < user_attributes.NumberOfRows)
				{
					IList<int> attribute_list = user_attributes.GetEntriesByRow(u);
					if(attribute_list.Count > 0)
					{
						double sum = 0;
						double second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += main_demo[attribute_id];
						}
						prediction += sum / second_norm_denominator;
					}
				}
				
				for(int d = 0; d < additional_user_attributes.Count; d++)
				{
					if(u < additional_user_attributes[d].NumberOfRows)
					{
						IList<int> attribute_list = additional_user_attributes[d].GetEntriesByRow(u);
						if(attribute_list.Count > 0)
						{
							double sum = 0;
							double second_norm_denominator = attribute_list.Count;
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
						double sum = 0;
						double second_norm_denominator = attribute_list.Count;
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
							double sum = 0;
							double second_norm_denominator = attribute_list.Count;
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
					double item_norm_denominator = item_attribute_list.Count;
					
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
				
				prediction += DataType.MatrixExtensions.RowScalarProduct(user_factors, u, item_factors, i);

				double err = ratings[index] - prediction;

				float user_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByUser[u])) : reg;
				float item_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByItem[i])) : reg;

				// adjust biases
				if (update_user)
					user_bias[u] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * user_reg_weight * user_bias[u]);
				if (update_item)
					item_bias[i] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * item_reg_weight * item_bias[i]);
				
				// adjust user attributes
				if(u < user_attributes.NumberOfRows)
				{
					IList<int> attribute_list = user_attributes.GetEntriesByRow(u);
					if(attribute_list.Count > 0)
					{
						double second_norm_denominator = attribute_list.Count;
						double second_norm_error = err / second_norm_denominator;

						foreach (int attribute_id in attribute_list)
						{							
							main_demo[attribute_id] += BiasLearnRate * current_learnrate * ((float) second_norm_error - BiasReg * reg * main_demo[attribute_id]);
						}
					}
				}
				
				for(int d = 0; d < additional_user_attributes.Count; d++)
				{
					if(u < additional_user_attributes[d].NumberOfRows)
					{
						IList<int> attribute_list = additional_user_attributes[d].GetEntriesByRow(u);
						if(attribute_list.Count > 0)
						{
							double second_norm_denominator = attribute_list.Count;
							double second_norm_error = err / second_norm_denominator;
	
							foreach (int attribute_id in attribute_list)
							{							
								second_demo[d][attribute_id] += BiasLearnRate * current_learnrate * ((float) second_norm_error - BiasReg * reg * second_demo[d][attribute_id]);
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
					
					float norm_error = (float)err / item_norm_denominator;
					
					foreach(int u_att in user_attribute_list)
					{
						foreach(int i_att in item_attribute_list)
						{
							h[0][u_att, i_att] += BiasLearnRate * current_learnrate * (norm_error * user_norm - BiasReg * reg * h[0][u_att, i_att]);
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
								h[d + 1][u_att, i_att] += BiasLearnRate * current_learnrate * (norm_error * user_norm - BiasReg * reg * h[d + 1][u_att, i_att]);;
							}
						}									
					}
				}
				
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
			}

			UpdateLearnRate();
		}
		
		///
		public override float Predict(int user_id, int item_id)
		{
			double result = global_bias;

			if (user_id < user_bias.Length)
				result += user_bias[user_id];
			if (item_id < item_bias.Length)
				result += item_bias[item_id];
			
			if(user_id < user_attributes.NumberOfRows)
			{
				IList<int> attribute_list = user_attributes.GetEntriesByRow(user_id);
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
			
			for(int d = 0; d < additional_user_attributes.Count; d++)
			{
				if(user_id < additional_user_attributes[d].NumberOfRows)
				{
					IList<int> attribute_list = additional_user_attributes[d].GetEntriesByRow(user_id);
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
			
			if (user_id <= MaxUserID && item_id <= MaxItemID)
				result += DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);

			if (result > MaxRating)
				return MaxRating;
			if (result < MinRating)
				return MinRating;

			return (float) result;
		}
	}
}

